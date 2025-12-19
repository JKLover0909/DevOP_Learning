import argparse
import pickle
from collections import namedtuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

import gymnasium as gym # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.distributions import Normal # type: ignore
from torch.distributions import Categorical # type: ignore
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler # type: ignore

import place_env
import torchvision # type: ignore
from torchvision.models import ResNet18_Weights
from place_db import PlaceDB
import time
from tqdm import tqdm # type: ignore
import random
from comp_res import comp_res
from torch.utils.tensorboard import SummaryWriter # type: ignore 

# set device to cpu or cuda
device = torch.device('cuda')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--soft_coefficient', type=float, default = 1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--save_fig', action='store_true', default=False)
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to model checkpoint (.pkl file) for testing or resuming training')
parser.add_argument('--ordering_method', type=str, default='default',
                    choices=['default', 'pagerank', 'eigenvector', 'degree'],
                    help='Method to order macro placement: default (topology), pagerank, eigenvector, degree')
parser.add_argument('--centrality_file', type=str, default=None,
                    help='Path to precomputed centrality pickle file (optional)')
parser.add_argument('--keep_top_n', type=int, default=5,
                    help='Number of top models to keep (default: 5, set to 0 to keep all)')
args = parser.parse_args()
writer = SummaryWriter('./tb_log')

benchmark = args.benchmark
ordering_method = args.ordering_method
placedb = PlaceDB(benchmark, ordering_method=ordering_method, centrality_file=args.centrality_file)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb = placedb, placed_num_macro = placed_num_macro, grid = grid, disable_env_checker=True).unwrapped

num_emb_state = 64 + 2 + 1
num_state = 1 + grid*grid*5 + 2

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_action = env.action_space.shape
seed_torch(args.seed)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
print("seed = {}".format(args.seed))
print("lr = {}".format(args.lr))
print("placed_num_macro = {}".format(args.pnm))


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self, res_net):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)


class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid * grid)
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.gcn = None
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)

    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                        x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                        ),dim= 1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + args.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res, gcn_res


class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    if placed_num_macro:
        buffer_capacity = 10 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size
    print("batch_size = {}".format(batch_size))

    def __init__(self):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn,  cnn_coarse = None, res_net = self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models"):
            os.mkdir("save_models")
        
        # Save new model with ordering_method in filename
        model_filename = "net_dict-{}-{}-{}-{}-{}.pkl".format(
            benchmark, ordering_method, placed_num_macro, strftime, int(running_reward)
        )
        model_path = os.path.join("./save_models", model_filename)
        torch.save({
            "actor_net_dict": self.actor_net.state_dict(),
            "critic_net_dict": self.critic_net.state_dict(),
            "reward": running_reward,
            "benchmark": benchmark,
            "ordering_method": ordering_method,
            "pnm": placed_num_macro
        }, model_path)
        print(f"Saved model: {model_filename}")
        
        # Keep only top N models if keep_top_n > 0
        if args.keep_top_n > 0:
            self._cleanup_old_models(args.keep_top_n)
    
    def _cleanup_old_models(self, keep_top_n):
        """Keep only top N models by reward, delete others for same benchmark/ordering/pnm"""
        import glob
        import re
        
        # Find all model files for current benchmark, ordering_method and pnm
        pattern = f"./save_models/net_dict-{benchmark}-{ordering_method}-{placed_num_macro}-*.pkl"
        model_files = glob.glob(pattern)
        
        if len(model_files) <= keep_top_n:
            return  # Not enough models to clean up
        
        # Parse reward from filename: net_dict-{benchmark}-{ordering}-{pnm}-{timestamp}--{reward}.pkl
        # Note: reward has double dash (--) before it, representing negative numbers
        models_with_reward = []
        for filepath in model_files:
            try:
                filename = os.path.basename(filepath)
                # Use regex to extract reward with proper sign handling
                # Match pattern: --{digits}.pkl at end of filename
                match = re.search(r'--(-?\d+)\.pkl$', filename)
                if match:
                    reward = int(match.group(1))
                    # Reward in filename is already negative (stored as --54554 meaning -54554)
                    # The double dash is: one from timestamp separator + one from negative sign
                    reward = -abs(reward)  # Ensure it's negative
                    models_with_reward.append((filepath, reward))
            except (ValueError, IndexError, AttributeError):
                # Skip files that don't match expected format
                continue
        
        # Sort by reward (descending, closer to 0 is better)
        # For negative numbers: -100 > -1000 > -10000
        models_with_reward.sort(key=lambda x: x[1], reverse=True)
        models_to_keep = set([m[0] for m in models_with_reward[:keep_top_n]])
        
        # Delete models not in top N
        deleted_count = 0
        for filepath, reward in models_with_reward[keep_top_n:]:
            try:
                os.remove(filepath)
                deleted_count += 1
                print(f"Deleted old model (reward={reward}): {os.path.basename(filepath)}")
            except OSError as e:
                print(f"Warning: Failed to delete {filepath}: {e}")
        
        if deleted_count > 0:
            print(f"Kept top {keep_top_n} models, deleted {deleted_count} old models")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + args.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = args.disable_tqdm):
                self.training_step +=1
                
                action_probs, _, _ = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]                
                critic_net_output = self.critic_net(state[index].to(device))
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                writer.add_scalar('action_loss', action_loss, self.training_step)
                writer.add_scalar('value_loss', value_loss, self.training_step)


def save_placement(file_path, node_pos, ratio):
    fwrite = open(file_path, 'w')
    node_place = {}
    for node_name in node_pos:

        x, y,_ , _ = node_pos[node_name]
        x = round(x * ratio + ratio) 
        y = round(y * ratio + ratio)
        node_place[node_name] = (x, y)
    print("len node_place", len(node_place))
    for node_name in placedb.node_info:
        if node_name not in node_place:
            continue
        x, y = node_place[node_name]
        fwrite.write('{}\t{}\t{}\t:\tN /FIXED\n'.format(node_name, x, y))
    print(".pl has been saved to {}.".format(file_path))


def main():

    agent = PPO()
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    training_records = []
    running_reward = -1000000
    

    log_file_name = "logs/log_"+ benchmark + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
    if not os.path.exists("logs"):
        os.mkdir("logs")
    fwrite = open(log_file_name, "w")
    
    # Load model if specified
    load_model_path = args.model_path
    
    # If testing mode but no model specified, try to find the best model
    if args.is_test and load_model_path is None:
        if os.path.exists("save_models"):
            # Find all models for this benchmark, ordering_method and pnm
            model_files = [f for f in os.listdir("save_models") 
                          if f.startswith(f"net_dict-{benchmark}-{ordering_method}-{placed_num_macro}") and f.endswith(".pkl")]
            if model_files:
                # Sort by reward (filename ends with reward value before .pkl)
                def get_reward_from_filename(filename):
                    try:
                        # Extract reward from filename like: net_dict-adaptec1-pagerank-128-2024-12-16-10-30-45-12345.pkl
                        reward_str = filename.split('-')[-1].replace('.pkl', '')
                        return int(reward_str)
                    except:
                        return float('-inf')
                
                model_files.sort(key=get_reward_from_filename, reverse=True)
                load_model_path = os.path.join("save_models", model_files[0])
                print(f"\n{'='*60}")
                print(f"TEST MODE: Auto-loading best model:")
                print(f"  {load_model_path}")
                print(f"  Ordering: {ordering_method}")
                print(f"  Reward: {get_reward_from_filename(model_files[0])}")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"WARNING: TEST MODE but no saved models found!")
                print(f"  Looking for: save_models/net_dict-{benchmark}-{ordering_method}-{placed_num_macro}-*.pkl")
                print(f"  Will use randomly initialized model (not recommended)")
                print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"WARNING: TEST MODE but save_models directory not found!")
            print(f"  Will use randomly initialized model (not recommended)")
            print(f"{'='*60}\n")
    
    if load_model_path:
        if os.path.exists(load_model_path):
            print(f"Loading model from: {load_model_path}")
            agent.load_param(load_model_path)
            print(f"Model loaded successfully!")
        else:
            print(f"ERROR: Model file not found: {load_model_path}")
            print(f"Exiting...")
            exit(1)
    
    best_reward = running_reward
    best_reward_ever = running_reward  # Track absolute best for early stopping
    epochs_without_improvement = 0  # Counter for early stopping
    best_hpwl = float('inf')  # Track best HPWL for --is_test mode
    if args.is_test:
        torch.inference_mode()

    for i_epoch in range(100000):
        score = 0
        raw_score = 0
        start = time.time()
        state = env.reset()

        done = False
        while done is False:
            state_tmp = state.copy()
            action, action_log_prob = agent.select_action(state)
        
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (num_state, )
            reward_intrinsic = 0
            if not args.is_test:
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
            if not args.is_test and agent.store_transition(trans):                
                assert done == True
                agent.update()
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
        end = time.time()

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print("score = {}, raw_score = {}".format(score, raw_score))

        # Check if this is the best reward ever (for early stopping)
        if running_reward > best_reward_ever:
            best_reward_ever = running_reward
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1
        
        if running_reward > best_reward * 0.975:
            best_reward = running_reward
            if i_epoch >= 10:
                agent.save_param(running_reward)
                if args.save_fig:
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    if not os.path.exists("figures"):
                        os.mkdir("figures")
                    env.save_fig("./figures/{}{}.png".format(strftime_now,int(raw_score)))
                    print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                    print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
                except:
                    assert False
        
        if args.is_test:
            print("save node_pos")
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
            print("time = {}s".format(end-start))
            
            # Only save .pl if HPWL is better than previous best (overwrite old file)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                out_dir = os.path.join(os.getcwd(), 'gg_place_new')
                os.makedirs(out_dir, exist_ok=True)
                
                # Single best .pl file with fixed name (ghi đè file cũ)
                pl_best_path = os.path.join(out_dir, f'{benchmark}-best.pl')
                print(f"Saving best placement to {pl_best_path} (HPWL={hpwl:.2f})")
                
                fwrite_pl = open(pl_best_path, 'w')
                for node_name in env.node_pos:
                    if node_name == "V":
                        continue
                    x, y, size_x, size_y = env.node_pos[node_name]
                    x = x * env.ratio + placedb.node_info[node_name]['x'] /2.0
                    y = y * env.ratio + placedb.node_info[node_name]['y'] /2.0
                    fwrite_pl.write("{}\t{:.4f}\t{:.4f}\n".format(node_name, x, y))
                fwrite_pl.close()
                
                # Save figure for best result
                os.makedirs("figures", exist_ok=True)
                env.save_fig(f"./figures/{benchmark}-best.png")
                print(f"Saved best figure to ./figures/{benchmark}-best.png")
        
        training_records.append(TrainingRecord(i_epoch, running_reward))
        if i_epoch % 1 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
            fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
            fwrite.flush()
        writer.add_scalar('reward', running_reward, i_epoch)
        
        # Early stopping: No improvement for 100 epochs
        if epochs_without_improvement >= 100:
            print("\n" + "="*60)
            print("Early stopping triggered!")
            print(f"No improvement for {epochs_without_improvement} epochs.")
            print(f"Best reward achieved: {best_reward_ever:.2f}")
            print(f"Current reward: {running_reward:.2f}")
            print("Stopping training...")
            print("="*60 + "\n")
            env.close()
            break
        
        # Convergence check
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param() # type: ignore
            break
        if i_epoch % 100 == 0:
            if placed_num_macro is None:
                env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))

        
if __name__ == '__main__':
    main()
