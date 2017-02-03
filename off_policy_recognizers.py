import numpy as np
import matplotlib.pyplot as plt

# 3x3 Grid World with '1' being a Terminal state
states = {'1','2','3',
          '4','5','6',
          '7','8','9'}

actions = {'N','E','S','W'}

reward = { '1':0,'2':-1,'3':-1,
          '4':-1,'5':-1,'6':-1,
          '7':-1,'8':-1,'9':-1}

# target policy no recognizer
tar_norecog={'N':0.55,'E':0,'S':0,'W':0.45}

# target policy with recognizer
tar_recog = {'N':5/7,'E':0,'W':2/7,'S':0}

# behavior policy
mu_b={'N':0.7,'E':0.01,'S':0.01,'W':0.28}

# recognizer
c={'N':1,'E':0,'S':0,'W':1}

possible_moves={'8':{'N':'5','E':'9','W':'7','S':'8'},
                '7':{'N':'4','E':'8','S':'7','W':'7'},
                '9':{'N':'6','E':'9','S':'9','W':'8'},
                '4':{'N':'1','E':'5','S':'7','W':'4'},
                '5':{'N':'2','E':'6','S':'8','W':'4'},
                '6':{'N':'3','E':'6','S':'9','W':'5'},
                '2':{'N':'2','E':'3','S':'5','W':'1'},
                '3':{'N':'3','E':'3','S':'6','W':'2'}
}

# generates an action in grid world
# returns new state, old state, and action
def gen_action(s,m):
    a=np.random.uniform(0,1)
    if a <= m['N']:
        new_state=possible_moves[s]['N']
        action='N'
    elif m['N']<a<=m['N']+m['E']:
        new_state=possible_moves[s]['E']
        action='E'
    elif m['N']+m['E']<a<=m['N']+m['E']+m['W']:
        new_state=possible_moves[s]['W']
        action='W'
    else:
        new_state=possible_moves[s]['S']
        action='S'
    return new_state, s, action


# simulate n episodes following policy m
# returns values of ord_importance, rec_importance, and weights
def sim(n,m):
    V_ord_list=[]
    V_rec_list=[]
    rho_wei=[]
    for x in range(n):
        V=0
        rho_ord=1
        rho_rec=1
        i_state='8'
        while i_state!='1':
            state = gen_action(i_state,m)
            V += reward[state[0]]
            if state[1]!='8':
                mu_st='8'
            else:
                mu_st='8'
            rho_ord *= tar_norecog[state[2]]/m[state[2]]
            rho_rec *= c[state[2]]/(mu_b['N']+mu_b['W'])
            i_state=state[0]
        V_ord = V*rho_ord
        V_ord_list.append(V_ord)
        V_rec_list.append(V*rho_rec)
        rho_wei.append(rho_ord)
    return V_ord_list, V_rec_list , rho_wei

no_episodes=np.arange(50,410,10)
Variances_ord=[]
Variances_rec=[]
Variances_wei=[]
np.random.seed(3)
for n in no_episodes:
    Sample_ord_means=[]
    Sample_rec_means=[]
    Sample_wei_means=[]
    print(n)
    for x in range(200):
        v_pi=sim(n,mu_b)
        Sample_ord_means.append(np.mean(v_pi[0]))
        Sample_rec_means.append(np.mean(v_pi[1]))
        Sample_wei_means.append(sum(v_pi[0])/sum(v_pi[2]))
    Variances_rec.append(np.var(Sample_rec_means))
    Variances_ord.append(np.var(Sample_ord_means))
    Variances_wei.append(np.var(Sample_wei_means))

line1=plt.plot(no_episodes,Variances_ord,'b-', label='ordinary')
line2 = plt.plot(no_episodes,Variances_rec,'r-', label='recognizer')
line3= plt.plot(no_episodes,Variances_wei,'g-',label='weighted')
plt.xlabel('Number of Episodes')
plt.ylabel('Variance')
plt.title('Sample Variances after 200 runs')
plt.legend()
plt.show()
