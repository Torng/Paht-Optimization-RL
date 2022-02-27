import sys
from path_agent import PathAgent
from path_env import PathEnv
import torch
from preprocessor import Preprocessor
from Module.dqn import DQN
import torch.optim as optim
from itertools import count
import config
import matplotlib.pyplot as plt

# Cuda detect
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU State:', device)

# print(device); exit()
# Environment setting
preprocessor = Preprocessor()
preprocessor.process()
path_env = PathEnv(preprocessor.all_order, preprocessor.motors, preprocessor.actions, preprocessor.time_matrix,
                   preprocessor.distance_matrix)
test_env = PathEnv(preprocessor.all_order, preprocessor.motors, preprocessor.actions, preprocessor.time_matrix,
                   preprocessor.distance_matrix)
path_env.reset()
attribute_count = preprocessor.all_order.observation[0].get_observation().size

policy_net = DQN(len(preprocessor.job_info_list), attribute_count, len(path_env.ac), device).to(device)

optimizer = optim.adam(policy_net.parameters(), lr=config.LEARNING_RATE)
schedule_agent = PathAgent(policy_net, device, len(preprocessor.job_info_list), attribute_count,
                           len(path_env.ac), optimizer)

returns = []
best_return = config.BEST_RETURN
best_observation = {}
tolerance_step = config.TOLERANCE_STEP
tolerance_count = 0
total_step = 0
for i_episode in range(config.TRAINING_EPISODE):
    # Initialize the environment and state
    state = path_env.reset()
    for t in count():
        # Select and perform an action
        action = schedule_agent.select_action(state)
        total_step += 1
        _, reward, done = path_env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_state = path_env.get_current_state()
        schedule_agent.memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = schedule_agent.optimize_model()
        if total_step % config.LOG_INTERVAL == 0:
            print("step {0} --> loss : {1} ".format(str(total_step), str(loss.item())))
        if total_step % config.EVAL_INTERVAL == 0:
            avg_return = schedule_agent.test_agent(test_env, len(preprocessor.job_info_list),
                                                   preprocessor.equipments_list, False)
            if best_return < avg_return:
                best_return = avg_return
                tolerance_count = 0
                # Store good results
                schedule_agent.save_best_observation(test_env, preprocessor.equipments_list, preprocessor.job_info_list)
                # Save model
                torch.save(schedule_agent.policy_net.state_dict(), config.MODEL_FILE)
            else:
                tolerance_count += 1
                if tolerance_count == tolerance_step:
                    break

            returns.append(avg_return)
            print("step {0} --> avg_return : {1}".format(str(total_step), str(avg_return)))
        if total_step % 5000 == 0:
            # test_agent(agent.policy, test_env)
            schedule_agent.test_agent(test_env, len(preprocessor.job_info_list), preprocessor.equipments_list)

            # Plot
            plt.plot(returns)
            plt.savefig('return.png')
            plt.close()
        if done:
            if all([ob.is_done for ob in path_env.job_info.observation]):
                schedule_agent.save_best_observation(test_env, preprocessor.equipments_list, preprocessor.job_info_list)
                finish_schedule_result = schedule_agent.best_observation[-1]
                finish_schedule_result = finish_schedule_result
                schedule_agent.test_agent(test_env, len(preprocessor.job_info_list), preprocessor.equipments_list)
                ScheduleKPI(finish_schedule_result, preprocessor.equipments_list, preprocessor.job_info_list)
                torch.save(schedule_agent.policy_net.state_dict(), config.MODEL_FILE + '_all_job_done')
                sys.exit()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % config.TARGET_UPDATE_PERIOD == 0:
        schedule_agent.update_target_net()

print('Complete')

finish_schedule_result = schedule_agent.best_observation[-1]

# Print KPI score
print("\n\nFinish Schedule Result:")
# clear_result(finish_schedule_result, len(preprocessor.job_info_list), len(preprocessor.equipments_list))
# finish_schedule_result = finish_schedule_result
# ScheduleKPI(finish_schedule_result, preprocessor.equipments_list, preprocessor.job_info_list)
