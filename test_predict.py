import sys; sys.path.insert(0, '.')
from server.app import env, predict, PredictRequest
from warehouse_env.models import Action

env.reset(task='easy')
s = env.state()
print('agent1 pos:', s['robots']['agent1']['pos'])
print('item1 pos:', s['inventory']['item1'])
print('goal:', s['goal'])
print()

for i in range(25):
    req = PredictRequest(agent_id='agent1')
    result = predict(req)
    src = result['source']
    atype = result['action_type']
    adir = result.get('direction')
    print(f'Step {i+1}: [{src}] -> {atype} {adir}')

    action = Action(agent_id='agent1', action_type=atype, direction=adir)
    obs, rew, done, info = env.step(action)
    s2 = env.state()
    carrying = s2['robots']['agent1']['carrying']
    deliveries = len(s2['completed_orders'])
    print(f'         pos={s2["robots"]["agent1"]["pos"]} carrying={carrying} deliveries={deliveries}')
    if carrying:
        print('  *** ITEM PICKED UP ***')
    if deliveries:
        print('  *** DELIVERY MADE ***')
    if done:
        print('DONE!')
        break
