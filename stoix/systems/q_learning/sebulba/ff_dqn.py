import copy
import queue
import threading
import warnings
from collections import defaultdict
from queue import Queue
from typing import Callable, Dict, List, Sequence, Tuple

from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    CoreOffPolicyLearnerState,
    CriticApply,
    ExperimentOutput,
    Observation,
    OnlineAndTarget,
    SebulbaLearnerFn,
)
from stoix.evaluator import get_distribution_act_fn, get_sebulba_eval_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.systems.q_learning.dqn_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import q_learning
from stoix.utils.rate_limiters import MinSize
from stoix.utils.sebulba_utils import (
    OffPolicyPipeline,
    ParamsSource,
    RecordTimeTo,
    ThreadLifetime,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_act_fn(
    q_apply_fn: ActorApply,
) -> Callable[
    [OnlineAndTarget, Observation, chex.PRNGKey], Tuple[chex.Array, chex.Array, chex.Array]
]:
    """Get the act function that is used by the actor threads."""

    def actor_fn(
        params: OnlineAndTarget, observation: Observation, rng_key: chex.PRNGKey
    ) -> chex.Array:
        rng_key, policy_key = jax.random.split(rng_key)
        pi = q_apply_fn(params.online, observation)
        action = pi.sample(seed=policy_key)
        return action

    return actor_fn


def get_rollout_fn(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: OffPolicyPipeline,
    q_apply_fn: ActorApply,
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
) -> Callable[[chex.PRNGKey], None]:
    """Get the rollout function that is used by the actor threads."""
    # Unpack and set up the functions
    act_fn = get_act_fn(q_apply_fn)
    act_fn = jax.jit(act_fn, device=actor_device)
    cpu = jax.devices("cpu")[0]
    move_to_device = lambda tree: jax.tree.map(lambda x: jax.device_put(x, actor_device), tree)
    split_key_fn = jax.jit(jax.random.split, device=actor_device)
    # Build the environments
    envs = env_factory(config.arch.actor.envs_per_actor)

    # Create the rollout function
    def rollout_fn(rng_key: chex.PRNGKey) -> None:
        # Ensure all computation is on the actor device
        with jax.default_device(actor_device):
            # Reset the environment
            timestep = envs.reset(seed=seeds)

            # Move the environment data to the actor device
            cached_obs = move_to_device(timestep.observation)

            # Loop until the thread is stopped
            while not thread_lifetime.should_stop():
                # Create the list to store transitions
                traj: List[Transition] = []
                # Create the dictionary to store timings for metrics
                timings_dict: Dict[str, List[float]] = defaultdict(list)
                # Rollout the environment
                with RecordTimeTo(timings_dict["single_rollout_time"]):
                    # Loop until the rollout length is reached
                    for _ in range(config.system.rollout_length):
                        # Get the latest parameters from the source
                        with RecordTimeTo(timings_dict["get_params_time"]):
                            params = params_source.get()

                        # Run the actor and critic networks to get the action, value and log_prob
                        with RecordTimeTo(timings_dict["compute_action_time"]):
                            rng_key, policy_key = split_key_fn(rng_key)
                            action = act_fn(params, cached_obs, policy_key)

                        # Move the action to the CPU
                        action_cpu = np.asarray(jax.device_put(action, cpu))

                        # Step the environment
                        with RecordTimeTo(timings_dict["env_step_time"]):
                            timestep = envs.step(action_cpu)

                        # Get the next dones and truncation flags
                        dones = np.logical_and(
                            np.asarray(timestep.last()), np.asarray(timestep.discount == 0.0)
                        )
                        # trunc = np.logical_and(
                        #     np.asarray(timestep.last()), np.asarray(timestep.discount == 1.0)
                        # )
                        cached_next_dones = move_to_device(dones)
                        # cached_next_trunc = move_to_device(trunc)
                        cached_next_obs = move_to_device(timestep.observation)

                        # Append PPOTransition to the trajectory list
                        reward = timestep.reward
                        metrics = timestep.extras["metrics"]

                        traj.append(
                            Transition(
                                cached_obs,
                                action,
                                reward,
                                cached_next_dones,
                                cached_next_obs,
                                metrics,
                            )
                        )
                        # Update the cached observations
                        cached_obs = cached_next_obs

                # Send the trajectory to the pipeline
                with RecordTimeTo(timings_dict["rollout_put_time"]):
                    try:
                        pipeline.put(traj, timings_dict)
                    except queue.Full:
                        warnings.warn(
                            "Waited too long to add to the rollout queue, killing the actor thread",
                            stacklevel=2,
                        )
                        break

            # Close the environments
            envs.close()

    return rollout_fn


def get_actor_thread(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: OffPolicyPipeline,
    q_apply_fn: ActorApply,
    rng_key: chex.PRNGKey,
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
    name: str,
) -> threading.Thread:
    """Get the actor thread that once started will collect data from the
    environment and send it to the pipeline."""
    rng_key = jax.device_put(rng_key, actor_device)

    rollout_fn = get_rollout_fn(
        env_factory,
        actor_device,
        params_source,
        pipeline,
        q_apply_fn,
        config,
        seeds,
        thread_lifetime,
    )

    actor = threading.Thread(
        target=rollout_fn,
        args=(rng_key,),
        name=name,
    )

    return actor


def get_learner_step_fn(
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> SebulbaLearnerFn[CoreOffPolicyLearnerState, Transition]:
    """Get the learner step function that is used by the learner thread to update the networks."""

    def _update_step(
        learner_state: CoreOffPolicyLearnerState, batch: Transition
    ) -> Tuple[CoreOffPolicyLearnerState, Tuple]:

        # Unpack learner state.
        params, opt_states, key = learner_state

        # Define the Q loss function.
        def _q_loss_fn(
            q_params: FrozenDict,
            target_q_params: FrozenDict,
            transitions: Transition,
        ) -> jnp.ndarray:

            q_tm1 = q_apply_fn(q_params, transitions.obs).preferences
            q_t = q_apply_fn(target_q_params, transitions.next_obs).preferences

            # Cast and clip rewards.
            discount = 1.0 - transitions.done.astype(jnp.float32)
            d_t = (discount * config.system.gamma).astype(jnp.float32)
            r_t = jnp.clip(
                transitions.reward, -config.system.max_abs_reward, config.system.max_abs_reward
            ).astype(jnp.float32)
            a_tm1 = transitions.action

            # Compute Q-learning loss.
            batch_loss = q_learning(
                q_tm1,
                a_tm1,
                r_t,
                d_t,
                q_t,
                config.system.huber_loss_parameter,
            )

            loss_info = {
                "q_loss": batch_loss,
            }

            return batch_loss, loss_info

        # CALCULATE Q LOSS AND GRADIENTS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            params.online,
            params.target,
            batch,
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

        # UPDATE Q PARAMS AND OPTIMISER STATE
        q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states)
        q_new_online_params = optax.apply_updates(params.online, q_updates)
        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            q_new_online_params, params.target, config.system.tau
        )
        q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

        # PACK LEARNER STATE
        learner_state = CoreOffPolicyLearnerState(q_new_params, q_new_opt_state, key)

        # PACK LOSS INFO
        loss_info = {
            **q_loss_info,
        }

        # TODO (edan): rework the way training metrics are returned
        metric = {}
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: CoreOffPolicyLearnerState, batch: TrajectoryBufferSample[Transition]
    ) -> ExperimentOutput[CoreOffPolicyLearnerState]:
        batch = batch.experience
        learner_state, (episode_info, loss_info) = _update_step(learner_state, batch)

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def get_learner_rollout_fn(
    learn_step: SebulbaLearnerFn[CoreOffPolicyLearnerState, Transition],
    config: DictConfig,
    eval_queue: Queue,
    pipeline: OffPolicyPipeline,
    params_sources: Sequence[ParamsSource],
) -> Callable[[CoreOffPolicyLearnerState], None]:
    """Get the learner rollout function that is used by the learner thread to update the networks.
    This function is what is actually run by the learner thread. It gets the data from the pipeline
    and uses the learner update function to update the networks. It then sends these intermediate
    network parameters to a queue for evaluation."""

    def learner_rollout(learner_state: CoreOffPolicyLearnerState) -> None:
        # Loop for the total number of evaluations selected to be performed.
        for _ in range(config.arch.num_evaluation):
            # Create the lists to store metrics and timings for this learning iteration.
            metrics: List[Tuple[Dict, Dict]] = []
            rollout_times: List[Dict] = []
            learn_timings: Dict[str, List[float]] = defaultdict(list)
            # Loop for the number of updates per evaluation
            for _ in range(config.arch.num_updates_per_eval):
                # Get the trajectory batch from the pipeline
                # This is blocking so it will wait until the pipeline has data.
                with RecordTimeTo(learn_timings["rollout_get_time"]):
                    traj_batch, rollout_time = pipeline.get()  # type: ignore

                # We then call the update function to update the networks
                with RecordTimeTo(learn_timings["learning_time"]):
                    learner_state, episode_metrics, train_metrics = learn_step(
                        learner_state, traj_batch
                    )

                # We store the metrics and timings for this update
                metrics.append((episode_metrics, train_metrics))
                rollout_times.append(rollout_time)

                # After the update we need to update the params sources with the new params
                unreplicated_params = unreplicate(learner_state.params)
                # We loop over all params sources and update them with the new params
                # This is so that all the actors can get the latest params
                for source in params_sources:
                    source.update(unreplicated_params)

            # We then pass all the environment metrics, training metrics, current learner state
            # and timings to the evaluation queue. This is so the evaluator correctly evaluates
            # the performance of the networks at this point in time.
            episode_metrics, train_metrics = jax.tree.map(lambda *x: np.asarray(x), *metrics)
            rollout_times = jax.tree.map(lambda *x: np.mean(x), *rollout_times)
            timing_dict = rollout_times | learn_timings
            timing_dict = jax.tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))
            try:
                # We add a timeout mainly for sanity checks
                # If the queue is full for more than 60 seconds we kill the learner thread
                # This should never happen
                eval_queue.put(
                    (episode_metrics, train_metrics, learner_state, timing_dict), timeout=60
                )
            except queue.Full:
                warnings.warn(
                    "Waited too long to add to the evaluation queue, killing the learner thread. "
                    "This should not happen.",
                    stacklevel=2,
                )
                break

    return learner_rollout


def get_learner_thread(
    learn: SebulbaLearnerFn[CoreOffPolicyLearnerState, Transition],
    learner_state: CoreOffPolicyLearnerState,
    config: DictConfig,
    eval_queue: Queue,
    pipeline: OffPolicyPipeline,
    params_sources: Sequence[ParamsSource],
) -> threading.Thread:
    """Get the learner thread that is used to update the networks."""

    learner_rollout_fn = get_learner_rollout_fn(learn, config, eval_queue, pipeline, params_sources)

    learner_thread = threading.Thread(
        target=learner_rollout_fn,
        args=(learner_state,),
        name="Learner",
    )

    return learner_thread


def learner_setup(
    env_factory: EnvFactory,
    keys: chex.Array,
    learner_devices: Sequence[jax.Device],
    config: DictConfig,
) -> Tuple[
    SebulbaLearnerFn[CoreOffPolicyLearnerState, Transition],
    Tuple[ActorApply, CriticApply],
    CoreOffPolicyLearnerState,
]:
    """Setup for the learner state and networks."""

    # Create a single environment just to get the observation and action specs.
    env = env_factory(num_envs=1)
    # Get number/dimension of actions.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim
    example_obs = env.observation_spec().generate_value()
    env.close()

    # PRNG keys.
    key, q_net_key = keys

    # Define networks and optimiser.
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )

    q_network = Actor(torso=q_network_torso, action_head=q_network_action_head)

    eval_q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
    )
    eval_q_network = Actor(torso=q_network_torso, action_head=eval_q_network_action_head)

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = example_obs
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise q params and optimiser state.
    q_online_params = q_network.init(q_net_key, init_x)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(q_online_params, q_target_params)
    opt_states = q_opt_state

    q_network_apply_fn = q_network.apply
    eval_q_network_apply_fn = eval_q_network.apply

    # Pack apply and update functions.
    apply_fns = (q_network_apply_fn, eval_q_network_apply_fn)
    update_fns = q_optim.update

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    assert config.system.total_batch_size % config.num_learning_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )

    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.total_buffer_size,
        min_length=config.system.total_batch_size,
        sample_batch_size=config.system.total_batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn_step = get_learner_step_fn(q_network_apply_fn, update_fns, config)
    learn_step = jax.pmap(learn_step, axis_name="device")

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(TParams=OnlineAndTarget)
        # Update the params
        params = restored_params

    # Define params to be replicated across learner devices.
    opt_states = q_opt_state
    replicate_learner = (params, opt_states)

    # Duplicate across learner devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=learner_devices)

    # Initialise learner state.
    params, opt_states = replicate_learner
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, len(learner_devices))
    init_learner_state = CoreOffPolicyLearnerState(params, opt_states, step_keys)

    return learn_step, apply_fns, init_learner_state, buffer_fns, buffer_states


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Perform some checks on the config
    # This additionally calculates certains
    # values based on the config
    config = check_total_timesteps(config)

    assert (
        config.arch.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate the number of updates per evaluation
    config.arch.num_updates_per_eval = int(config.arch.num_updates // config.arch.num_evaluation)

    # Get the learner and actor devices
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"
    # Extract the actor and learner devices
    actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [
        local_devices[device_id] for device_id in config.arch.learner.device_ids
    ]
    # For evaluation we simply use the first learner device
    evaluator_device = local_learner_devices[0]
    print(f"{Fore.BLUE}{Style.BRIGHT}Actors devices: {actor_devices}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Learner devices: {local_learner_devices}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Global devices: {global_devices}{Style.RESET_ALL}")
    # Set the number of learning and acting devices in the config
    # useful for keeping track of experimental setup
    config.num_learning_devices = len(local_learner_devices)
    config.num_actor_actor_devices = len(actor_devices)

    # Calculate the number of envs per actor
    assert (
        config.arch.num_envs == config.arch.total_num_envs
    ), "arch.num_envs must equal arch.total_num_envs for Sebulba architectures"
    # We first simply take the total number of envs and divide by the number of actor devices
    # to get the number of envs per actor device
    num_envs_per_actor_device = config.arch.total_num_envs // len(actor_devices)
    # We then divide this by the number of actors per device to get the number of envs per actor
    num_envs_per_actor = int(num_envs_per_actor_device // config.arch.actor.actor_per_device)
    config.arch.actor.envs_per_actor = num_envs_per_actor

    # We then perform a simple check to ensure that the number of envs per actor is
    # divisible by the number of learner devices. This is because we shard the envs
    # per actor across the learner devices This check is mainly relevant for on-policy
    # algorithms
    assert num_envs_per_actor % len(local_learner_devices) == 0, (
        f"The number of envs per actor must be divisible by the number of learner devices. "
        f"Got {num_envs_per_actor} envs per actor and {len(local_learner_devices)} learner devices"
    )

    # Create the environment factory.
    env_factory = environments.make_factory(config)
    assert isinstance(
        env_factory, EnvFactory
    ), "Environment factory must be an instance of EnvFactory"

    # PRNG keys.
    key, q_net_key, pipeline_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner.
    learn_step, apply_fns, learner_state, buffer_fns, buffer_states = learner_setup(
        env_factory, (key, q_net_key), local_learner_devices, config
    )
    q_apply_fn, eval_q_apply_fn = apply_fns
    eval_act_fn = get_distribution_act_fn(config, eval_q_apply_fn)
    # Setup evaluator.
    evaluator, evaluator_envs = get_sebulba_eval_fn(
        env_factory, eval_act_fn, config, np_rng, evaluator_device
    )

    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Get initial parameters
    initial_params = unreplicate(learner_state.params)

    # Get the number of steps per rollout
    steps_per_rollout = (
        config.system.rollout_length * config.arch.total_num_envs * config.arch.num_updates_per_eval
    )

    # Creating the pipeline
    # First we create the lifetime so we can stop the pipeline when we want
    pipeline_lifetime = ThreadLifetime()
    # Now we create the pipeline
    # TODO(edan): do this properly
    replay_buffer_add, replay_buffer_sample = buffer_fns
    rate_limiter = MinSize(1)
    pipeline = OffPolicyPipeline(
        replay_buffer_add,
        replay_buffer_sample,
        buffer_states,
        rate_limiter,
        pipeline_key,
        local_learner_devices,
        pipeline_lifetime,
    )
    # Start the pipeline
    pipeline.start()

    # Create a single lifetime for all the actors and params sources
    actors_lifetime = ThreadLifetime()
    params_sources_lifetime = ThreadLifetime()

    # Create the params sources and actor threads
    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    for actor_device in actor_devices:
        # Create 1 params source per actor device as this will be used
        # to pass the params to the actors
        params_source = ParamsSource(initial_params, actor_device, params_sources_lifetime)
        params_source.start()
        params_sources.append(params_source)
        # Now for each device we choose to create multiple actor threads
        for i in range(config.arch.actor.actor_per_device):
            key, actors_key = jax.random.split(key)
            seeds = np_rng.integers(
                np.iinfo(np.int32).max, size=config.arch.actor.envs_per_actor
            ).tolist()
            actor_thread = get_actor_thread(
                env_factory,
                actor_device,
                params_source,
                pipeline,
                q_apply_fn,
                actors_key,
                config,
                seeds,
                actors_lifetime,
                f"Actor-{actor_device}-{i}",
            )
            actor_thread.start()
            actor_threads.append(actor_thread)

    # Create the evaluation queue
    eval_queue: Queue = Queue(maxsize=config.arch.num_evaluation)
    # Create the learner thread
    learner_thread = get_learner_thread(
        learn_step, learner_state, config, eval_queue, pipeline, params_sources
    )
    learner_thread.start()

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e7)
    best_params = initial_params.online
    # This is the main loop, all it does is evaluation and logging.
    # Acting and learning is happening in their own threads.
    # This loop waits for the learner to finish an update before evaluation and logging.
    for eval_step in range(config.arch.num_evaluation):
        # Get the next set of params and metrics from the learner
        episode_metrics, train_metrics, learner_state, timings_dict = eval_queue.get(block=True)

        # Log the metrics and timings
        t = int(steps_per_rollout * (eval_step + 1))
        timings_dict["timestep"] = t
        logger.log(timings_dict, t, eval_step, LogEvent.MISC)

        # TODO (edan): rework the way training metrics are returned
        # episode_metrics, ep_completed = get_final_step_metrics(episode_metrics)
        # episode_metrics["steps_per_second"] = (
        #     steps_per_rollout / timings_dict["single_rollout_time"]
        # )
        # if ep_completed:
        #     logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluate the current model and log the metrics
        unreplicated_actor_params = unreplicate(learner_state.params.online)
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = evaluator(unreplicated_actor_params, eval_key)
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate(learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(unreplicated_actor_params)
            max_episode_return = episode_return

    evaluator_envs.close()
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing learner...{Style.RESET_ALL}")
    # Now we stop the learner
    learner_thread.join()

    # First we stop all actors
    actors_lifetime.stop()

    # Now we stop the actors and params sources
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing actors...{Style.RESET_ALL}")
    for actor in actor_threads:
        # We clear the pipeline before stopping each actor thread
        # since actors can be blocked on the pipeline
        pipeline.clear()
        actor.join()

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing pipeline...{Style.RESET_ALL}")
    # Stop the pipeline
    pipeline_lifetime.stop()
    pipeline.join()

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing params sources...{Style.RESET_ALL}")
    # Stop the params sources
    params_sources_lifetime.stop()
    for param_source in params_sources:
        param_source.join()

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_sebulba_eval_fn(
            env_factory, eval_act_fn, config, np_rng, evaluator_device, eval_multiplier=10
        )
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = abs_metric_evaluator(best_params, eval_key)

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/sebulba",
    config_name="default_ff_dqn.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}DQN experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
