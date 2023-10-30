import torch


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions, codes,
			 returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):

	# Critic's update
	for _ in range(50):
		values_pred = value_net(states)
		value_loss = (values_pred - returns).pow(2).mean()
		# weight decay
		for param in value_net.parameters():
			value_loss += param.pow(2).sum() * l2_reg
		optimizer_value.zero_grad()
		value_loss.backward()
		optimizer_value.step()

	# Policy's update
	log_probs = policy_net.get_log_prob(states, codes, actions)
	ratio = torch.exp(log_probs - fixed_log_probs)
	surr1 = ratio * advantages
	surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
	policy_surr = -torch.min(surr1, surr2).mean() + 0.5 * value_loss.item()
	optimizer_policy.zero_grad()
	policy_surr.backward()
	torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
	optimizer_policy.step()