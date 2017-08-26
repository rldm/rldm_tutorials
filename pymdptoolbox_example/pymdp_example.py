import mdptoolbox.example
import numpy as np

P, R = mdptoolbox.example.forest(p=0.3, r1=0.3, S=5)

prob = np.zeros((2, 5, 5))

prob[0] = [[0.3, 0.7, 0., 0., 0.],
           [0.3, 0.0, 0.7, 0., 0.],
           [0.3, 0.0, 0., 0.7, 0.],
           [0.3, 0.0, 0., 0., 0.7],
           [0.3, 0.0, 0., 0., 0.7]]

prob[1] = [[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.]]

rewards = np.zeros((5, 2))
rewards[0] = [0., 0.]
rewards[1] = [0., 1.]
rewards[2] = [0., 1.]
rewards[3] = [0., 1.]
rewards[4] = [0.3, 2.]

print(R)

vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()
print("")
print(vi.policy)
print(vi.V)

vi1 = mdptoolbox.mdp.ValueIteration(prob, rewards, 0.9)
vi1.run()
print("")
print(vi1.policy)
