import numpy as np
import torch
import os
import time
#import dmc2gym
import wandb
import utils
import utils_soda
from rich.console import Console
from video import VideoRecorder
from config import CFG
from curl_recon_loss import CurlSacAgent
from utils import Config
from augmentations import center_crop_image, random_conv, random_crop  
from visualize import visualize_tsne
from collections import Counter
from env.wrappers import make_env

config = Config.from_json(CFG)
WB_LOG = True
console = Console()

transforms = {
    "random_crop": random_crop,
    # "random_shift": random_shift,
    "random_conv": random_conv,
    "center_crop_image": center_crop_image,
}

if config.params.seed == -1:
    config.params.__dict__["seed"] = np.random.randint(1, 1000000)
    console.log("random seed value", config.params.seed)
utils.set_seed_everywhere(config.params.seed)


def make_directory():
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = f"{config.env.domain_name}-{config.env.task_name}"
    exp_name = (
        (
            (
                f"{env_name}-{ts}-im{str(config.env.image_size)}-b"
                + str(config.train.batch_size)
            )
            + "-s"
        )
        + str(config.params.seed)
        + "-"
    ) + config.encoder.type

    config.params.work_dir = f"{config.params.work_dir}/{exp_name}"

    utils.make_dir(config.params.work_dir)
    video_dir = utils.make_dir(os.path.join(config.params.work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(config.params.work_dir, "model"))
    buffer_dir = utils.make_dir(os.path.join(config.params.work_dir, "buffer"))
    aug_dir = utils.make_dir(os.path.join(config.params.work_dir, "augmentated_obs"))

    return video_dir, model_dir, buffer_dir, aug_dir


def make_agent(obs_shape, action_shape, config, device, WB_LOG):
    if config.train.agent == "curl_sac":
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            WB_LOG=WB_LOG,
            hidden_dim=config.train.hidden_dim,
            discount=config.sac.discount,
            init_temperature=config.sac.init_temp,
            alpha_lr=config.sac.alpha_lr,
            alpha_beta=config.sac.alpha_beta,
            actor_lr=config.actor.lr,
            actor_beta=config.actor.beta,
            actor_log_std_min=config.actor.log_std_min,
            actor_log_std_max=config.actor.log_std_max,
            actor_update_freq=config.actor.update_freq,
            critic_lr=config.critic.lr,
            critic_beta=config.critic.beta,
            critic_tau=config.critic.tau,
            critic_target_update_freq=config.critic.target_update_freq,
            encoder_type=config.encoder.type,
            encoder_feature_dim=config.encoder.feature_dim,
            encoder_lr=config.encoder.lr,
            encoder_tau=config.encoder.tau,
            decoder_type=config.decoder.type,
            decoder_lr=config.decoder.lr,
            decoder_update_freq=config.decoder.update_freq,
            decoder_latent_lambda=config.decoder.latent_lambda,
            decoder_weight_lambda=config.decoder.weight_lambda,
            num_layers=config.encoder.num_layers,
            num_filters=config.encoder.num_filters,
            log_interval=config.params.log_interval,
            detach_encoder=config.params.detach_encoder,
            curl_latent_dim=config.params.curl_latent_dim,
        )
    else:
        assert f"agent is not supported: {config.train.agent}"
    


def evaluate(env_eval, agent, video, num_episodes, step, device):
    all_ep_rewards = []
    embed = []
    np_embed = []
    actions = []
    labels = []
    def run_eval_loop(sample_stochastically=True):
        for i in range(num_episodes):
            obs = env_eval.reset()
            ctr = 0
            #video.init(enabled=(i == 0))
            video.init(enabled=True)
            done = False
            episode_reward = 0
            while not done:
                ctr +=1
                # center crop image
                if config.encoder.type == "pixel":
                    obs = transforms["center_crop_image"](obs, config.env.image_size)
                    #print("eval loop obs shape", obs.shape, type(obs))
                    obses = torch.as_tensor(obs, device=device).float().unsqueeze(0)
                    #print("eval loop obses tensor shape", obses.shape)
                    #print("testing encoder")
                    #_, _, _, _, temp = agent.actor(obses, detach_encoder=True)
                    #print('shape of temp',temp.shape)
                    '''
                    # Uncomment the below lines to collect embeddings of last 20 frames from each episode.
                    if ctr > 105: # set ctr to 105 for cartpole and walker walk, 230 for cheetah
                        obses = torch.as_tensor(obs, device=device).float().unsqueeze(0)
                        _, _, _, _, temp = agent.actor(obses, detach_encoder=True)
                        #print(agent.actor.encoder)
                        #print(temp.shape, type(temp))
                        embed.append(temp.cpu().detach().numpy())
                    '''
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        '''
                        # Uncomment the below lines to collect the actions corresponding to the above collected embeddings.
                        if ctr > 105: # set ctr to 105 for cartpole and walker walk, 230 for cheetah
                            action = agent.select_action(obs)
                            #print("type of action",type(action))
                            #tmp = float("{:.5f}".format(action.item())) 
                            #print(action)
                            tmp = float(action.item())
                            actions.append(tmp)
                        '''
                        action = agent.select_action(obs)
                obs, reward, done, _ = env_eval.step(action)
                video.record(env_eval)
                episode_reward += reward
            #print("lenght of embed", len(embed))
            #print(f"length of embed after episode {i}: ", len(embed), len(embed[0][0]))
            #print("type of embed", type(embed), type(embed[0]))
            
            video.save(f"{step}_{i}.mp4")
            if WB_LOG:
                wandb.log({"eval/episode_reward": episode_reward, "step": step})
            all_ep_rewards.append(episode_reward)
        

        
        # t-SNE plots
        #print("after 10 eval episodes", len(embed), type(embed))
        #np_embed = np.array(embed)
        #print("after 10 eval episodes and conversion", np_embed.shape, type(np_embed))
        #x_train = np.reshape(np_embed, [np_embed.shape[0], np_embed.shape[1]*np_embed.shape[2]])
        #print("after 10 eval episodes and conversion and reshaping", x_train.shape, type(x_train))
        #print("labels", labels.shape, type(labels))
        #print(f"action labels at step : {step}", len(actions))
        #print("Count unique in actions", Counter(actions))

        '''
        # please note that this part is no longer needed as the labels were generated using k means clustering
        for i, val in enumerate(actions):
            if -1.0<= val <-0.5:
                labels.append(0)
            elif -0.5 <= val < 0:
                labels.append(1)
            elif 0<=val<0.5:
                labels.append(2)
            else:
                labels.append(3)
        '''
        # call visualize function
        #visualize_tsne(x_train, labels, step)

        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        console.log(
            f"Eval  | Step- {step}, Mean episode reward- {mean_ep_reward:.4f}, Best episode reward- {best_ep_reward:.4f}"
        )
        if WB_LOG:
            wandb.log({"eval/mean_episode_reward": mean_ep_reward, "step": step})
            wandb.log({"eval/best_episode_reward": best_ep_reward, "step": step})
        
        
    run_eval_loop(sample_stochastically=False)


def train(env, env_eval, agent, video, model_dir, replay_buffer, buffer_dir, device):
    c1 = config.params.c1
    c2 = config.params.c2
    c3 = config.params.c3
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    mean_all_episode_rewards = []
    for step in range(config.train.num_train_steps):
        # evaluate agent periodically
        if step % config.eval.eval_freq == 0:
            if WB_LOG:
                wandb.log({"eval/episode": episode, "step": step})
            evaluate(env_eval, agent, video, config.eval.num_eval_episodes, step, device)
            if config.params.save_model:
                agent.save_curl(model_dir, step)
            if config.params.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            mean_all_episode_rewards.append(episode_reward)
            if step > 0 and step % config.params.log_interval == 0 and WB_LOG:
                wandb.log({"train/duration": time.time() - start_time, "step": step})
                start_time = time.time()
            if step % config.params.log_interval == 0 and WB_LOG:
                wandb.log({"train/episode_reward": episode_reward, "step": step})
                wandb.log(
                    {
                        "train/mean_all_episode_reward": np.mean(
                            mean_all_episode_rewards
                        ),
                        "step": step,
                    }
                )

            console.log(
                f"Train | Episode- {episode},Step- {step}, Episode reward- {episode_reward:.4f}, Duration- {time.time()-start_time:.4f}"
            )

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % config.params.log_interval == 0 and WB_LOG:
                wandb.log({"train/episode": episode, "step": step})

        # sample action for data collection
        if step < config.train.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= config.train.init_steps:
            for _ in range(config.train.num_updates):
                agent.update(replay_buffer, step, WB_LOG, c1, c2, c3)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


def main():

    if WB_LOG:
        print("WandB version", wandb.__version__)
        wandb.login()
        wandb.init(project=config.env.domain_name, config=CFG)

    if config.env.transform == "random_conv":
        pre_transform_image_size = config.env.image_size
    else:
    	pre_transform_image_size = config.env.pre_transform_image_size
        
    env = make_env(
        domain_name=config.env.domain_name,
        task_name=config.env.task_name,
        seed=config.params.seed,
        #visualize_reward=False,
        #from_pixels=(config.encoder.type == "pixel"),
        #height=pre_transform_image_size,
        #width=pre_transform_image_size,
        episode_length=config.env.episode_length,
        frame_stack = 3,
        action_repeat=config.env.action_repeat,
        image_size=config.env.pre_transform_image_size,
        mode='train'
	   )
    env.seed(config.params.seed)
	
    test_env = make_env(
	    domain_name=config.env.domain_name,
	    task_name=config.env.task_name,
	    seed=config.params.seed,
	    #visualize_reward=False,
	    #from_pixels=(config.encoder.type == "pixel"),
	    #height=config.env.eval_pre_transform_image_size,
	    #width=config.env.eval_pre_transform_image_size,
	    episode_length=config.env.episode_length,
	    frame_stack = config.env.frame_stack,
	    action_repeat=config.env.action_repeat,
	    image_size=config.env.eval_pre_transform_image_size,
	    mode=config.eval.mode,
        ) if config.eval.mode is not None else None

   
    test_env.seed(config.params.seed)

    action_shape = env.action_space.shape

    if config.encoder.type == "pixel":
        obs_shape = (
            3 * config.env.frame_stack,
            config.env.image_size,
            config.env.image_size,
        )
        pre_aug_obs_shape = (
            3 * config.env.frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )


   
    '''
    # stack several consecutive frames together
    if config.encoder.type == "pixel":
        env = utils.FrameStack(env, k=config.env.frame_stack)
        test_env = utils.FrameStack(env_eval, k=config.env.frame_stack)
    '''
	
    video_dir, model_dir, buffer_dir, aug_dir = make_directory()

    video = VideoRecorder(video_dir if config.params.save_video else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=config.replay_buffer.capacity,
        batch_size=config.train.batch_size,
        device=device,
        image_size=config.env.image_size,
        transform=transforms[config.env.transform],
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        config=config,
        device=device,
        WB_LOG=WB_LOG,
    )
	
    #print("details of agent-")
    #print(agent)
    train(env, test_env, agent, video, model_dir, replay_buffer, buffer_dir, device)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # torch.backends.cudnn.benchmark = False
    main()
