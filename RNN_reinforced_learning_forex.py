import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.distributions import Categorical, TransformedDistribution, TanhTransform
import matplotlib.pyplot as plt
STARTING_BALANCE = 300
CSV_DELIMETER = '\t'


# MACHINE LEARNING CONFIGURATION
ACTION_SPACE = 1
MIN_ACTION = 0
MAX_ACTION = STARTING_BALANCE
HIDDEN_SIZE = 1024
LR = 1e-4 #learning rate: jak szybko model reaguje na zmiany gradientu
GAMMA = 0.99 #im wiekszy przedzial czasowy danych trenujacych, tym blizsze musi być 1, zeby nagroda koncowa nie rozmyla sie przy jej propagacji wstecznej
CLIP_EPS = 0.1
SAME_DATA_REUSES = 1
TRAINING_SESSIONS = 200 #do podpięcia różnorodnymi danymi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(profile = "full")

# TRADING ENVIRONMENT
class TradingEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop = True)
        self.steps_no = len(df)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.balance = STARTING_BALANCE
        self.entry_price = 0.0
        self.done = False
        self.total_profit = 0.0

    def step(self, action):
        if self.done:
            raise Exception("Epizod skończony. Środowisko do restartu.")
            
        self.step_idx = self.step_idx + 1
        if self.step_idx >= self.steps_no:
            self.done = True

        return self.done, self.step_idx

# --- PPO Actor-Critic with RNN ---
class PPO_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first = True)
        #self.norm = nn.LayerNorm(hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim)) 
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx):        
        out, hx = self.rnn(x, hx)
        #out = self.norm(out)
        mean = self.actor_mean(out)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(out)
        return mean, std, value, hx

def compute_returns_back_propagation(rewards, gamma = GAMMA):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype = torch.float32, device = device)

def ppo_update(model, optimizer, obs_seq, actions, old_log_probs, returns, advantages):
    for _ in range(SAME_DATA_REUSES):
        new_values = []
        new_log_probs = []
        hx = torch.zeros(1, 1, model.rnn.hidden_size, device = device)
        for t in range(len(obs_seq)):
            mean, std, value, hx = model(obs_seq[t], hx)
            dist = torch.distributions.Normal(mean, std)
            new_log_prob = dist.log_prob(actions[t])
            #dist = Categorical(logits = logits.squeeze(0).squeeze(0))
            new_values.append(value.squeeze(0).squeeze(0))
            new_log_probs.append(new_log_prob)

        new_values = torch.stack(new_values)
        new_log_probs = torch.stack(new_log_probs)
            
        entropy = dist.entropy().mean()
        ratio = (new_log_probs - old_log_probs.clone().detach()).exp() #drugi detach FIXING???!
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        #returns = (returns - returns.mean()) / (returns.std() + 1e-8) normalizacja przy danych malych nie ma sensu
        value_loss = (returns - new_values).pow(2).mean() #new_values?
        loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().mean().item()
                #print(f"{name}: grad norm = {grad_norm:.6f}")
            #else:
                #print(f"{name}: grad = None")
        optimizer.step()
        
#moznaby zostawic srodek action_space jako brak akcji -> jakby przeskalowanie drugie zrobic (iloczyn - bo ewentualnie obciecie clip)
def end_balance_reward(actions, file, step_i):
    balance = STARTING_BALANCE
    bought_lots, time, spread = 0, 0, 0
    for action in actions:
        action_squashed = torch.tanh(action)
        action = (action_squashed + 1) / 2 * (MAX_ACTION - MIN_ACTION) + MIN_ACTION
        spread = file["<SPREAD>"].iloc[time] * 0.00001 #musi zostac uzaleznione od walut
        scaled_action = action * (balance + bought_lots * (file["<CLOSE>"].iloc[time] - spread)) / STARTING_BALANCE
        if action > round((STARTING_BALANCE - 1)/ 2): #buy
            amount = ((file["<CLOSE>"].iloc[time] + spread) * balance) * (action - round((STARTING_BALANCE - 1)/ 2)) / round((STARTING_BALANCE - 1)/ 2)
            bought_lots += amount
            balance -= (file["<CLOSE>"].iloc[time] + spread) * amount
        elif action < round((STARTING_BALANCE - 1)/ 2): #sell
            amount = ((file["<CLOSE>"].iloc[time] - spread) * bought_lots) * (round((STARTING_BALANCE - 1)/ 2) - action ) / round((STARTING_BALANCE - 1)/ 2)
            bought_lots -= amount
            balance += (file["<CLOSE>"].iloc[time] - spread) * amount
        time += 1
    return 10000 * (balance + bought_lots * (file["<CLOSE>"].iloc[-1] - spread) - STARTING_BALANCE) / STARTING_BALANCE #todo: 100 byloby procentowo, 100*100 jest zeby funkcja celowa byla wyrazniejsza

def prep_observations(file, step_i, action_value, balance_value):
    observation_step = torch.tensor(file.iloc[step_i, 2:].astype(float).values, dtype = torch.float32, device = device)
    #observation_step[0:3] = observation_step[0:3] * 2 why?
    unsqueezed_observation = observation_step.unsqueeze(0).unsqueeze(0)
    unsqueezed_observation_with_action = torch.cat((unsqueezed_observation, torch.tensor([[[action_value]]], device = device)), dim = 2)
    unsqueezed_observation_with_action_and_balance = torch.cat((unsqueezed_observation_with_action, torch.tensor([[[balance_value]]], device = device)), dim = 2)
    return unsqueezed_observation_with_action_and_balance.float()

def train_ppo_and_test(csv_path):
    df = pd.read_csv(csv_path, delimiter = CSV_DELIMETER)
    data_cols = [col for col in df.columns if col != "<DATE>" and col != "<TIME>"]
    
    env = TradingEnv(df)

    model = PPO_RNN(input_dim = len(data_cols) + 2, hidden_dim = HIDDEN_SIZE, action_dim = ACTION_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr = LR)
    torch.autograd.set_detect_anomaly(True)
    results = []
    for episode in range(TRAINING_SESSIONS):
        obs_list, action_list, logprob_list, reward_list, value_list = [], [], [], [], []

        env.reset()
        done = False
        hx = None
        step_index, action, balance = 0, 0, 0
        
        while not done:
            unsqueezed_observation_with_action_and_balance = prep_observations(df, step_index, action, balance)
            mean, std, value, hx = model(unsqueezed_observation_with_action_and_balance, hx)
            dist = torch.distributions.Normal(mean, std)
            action_sample = dist.sample() 
            log_prob = dist.log_prob(action_sample)
            done, step_index = env.step(action_sample.item())

            obs_list.append(unsqueezed_observation_with_action_and_balance)
            action_list.append(action_sample)
            logprob_list.append(log_prob)
            value_list.append(value.squeeze(-1).clone())
            balance = end_balance_reward(action_list, df, step_index)
            reward_list.append(balance * (0.0 if not done else 1))

        logprob_list = torch.stack(logprob_list).unsqueeze(0)
        returns = compute_returns_back_propagation(reward_list).unsqueeze(0)
        #values = torch.zeros_like(returns, device = device)
        values = torch.stack(value_list).unsqueeze(0).detach() 
        advantages = returns - values
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalizacja przy danych malych nie ma sensu

        ppo_update(model, optimizer, obs_list, action_list, logprob_list, returns, advantages)
        results.append(reward_list[-1].item())
        print(f"[{episode+1}/{TRAINING_SESSIONS}] Profit: {reward_list[-1].item():.10f}")
    episodes = np.arange(1, len(results) + 1)

    coefficients = np.polyfit(episodes, results, deg = 1)
    trend_line = np.polyval(coefficients, episodes)

    plt.figure(figsize = (10, 5))
    plt.plot(episodes, results, marker = 'o', linestyle = '-', label = 'Profit')
    plt.plot(episodes, trend_line, color = 'red', linestyle = '--', label = 'Trend line')
    plt.axhline(y = sum(results) / len(results), color = 'green', linestyle = '-.', label = f'Average Profit')

    return model, plt

def ppo_test(model, csv_path, plot):
    df = pd.read_csv(csv_path, delimiter = CSV_DELIMETER)
    env = TradingEnv(df)

    env.reset()
    done = False
    hx = None
    step_index, action, balance = 0, 0, 0
    obs_list, action_list, reward_list = [], [], []

    while not done:
        unsqueezed_observation_with_action_and_balance = prep_observations(df, step_index, action, balance)
        mean, std, value, hx = model(unsqueezed_observation_with_action_and_balance, hx)
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.sample() 
        obs_list.append(unsqueezed_observation_with_action_and_balance)
        action_list.append(action_sample)
        done, _ = env.step(action)
        obs_list.append(unsqueezed_observation_with_action_and_balance)
        balance = end_balance_reward(action_list, df, step_index)
        reward_list.append(balance * (0.0 if not done else 1))


    plot.axhline(y = reward_list[-1].item(), color='yellow', linestyle='-.', label=f'Test after training')
    plt.title(f"Profit per Episode with Trend Line; LR: {LR}; HIDDEN_SIZE: {HIDDEN_SIZE}; STARTING_BALANCE: {STARTING_BALANCE}")
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_path = "forex_training_dataset_AUDCAD_M20_202507290000_202507300000.csv"
    test_path = "forex_test_dataset_AUDCAD_M20_202507300000_202507310000.csv"

    model, plot = train_ppo_and_test(train_path)
    ppo_test(model, test_path, plot)