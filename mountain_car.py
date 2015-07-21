import numpy as np

#if matplotlib is not present, plot will not be generated.
plot = True
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except:
    plot = False
import os

class MountainCar:
    mcar_min_position = -1.2
    mcar_max_position = 0.6
    mcar_max_velocity = 0.07
    mcar_min_velocity = -0.07
    mcar_goal_position=  0.5

    """
    This class implements the mountain car agent, as described in Sutton and Barto book, and in https://en.wikipedia.org/wiki/Mountain_Car.
    """

    def __init__(self,pos=-0.5,velocity=0):

        self.pos=pos
        self.velocity = velocity
        self.time_elapsed = 0

    def target_reached(self):
        return self.pos > self.mcar_goal_position


    def move(self,action):
        """
        Move the car according to an action.
        Action can be one of -1,0,1 or "reverse", "neutral","forward"
        """

        import math
        gravity = 0.0025 #originally, 0.0025
        thrust = 0.001 #originally, 0.001

        if action == "reverse": action = -1
        if action == "neutral": action = 0
        if action == "forward": action = 1
        assert ((action == -1) or (action == 0) or (action == 1))
        self.velocity+= (action)*thrust + math.cos(3*self.pos)*(-gravity)
        self.velocity = min(self.velocity,self.mcar_max_velocity)
        self.velocity = max(self.velocity,-1.0 * self.mcar_max_velocity)
        self.pos+=self.velocity
        if (self.pos < self.mcar_min_position):
            self.pos = self.mcar_min_position
            self.velocity = 0

        if (self.pos > self.mcar_max_position):
            self.pos = self.mcar_max_position
            self.velocity = 0

import imp


"""
MountainCarSarsaLearn and MountainCarQLearn are two algorithms implemented according to Sutton and Barto's book, chapter 8.4 (http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html)
"""
class MountainCarSarsaLearn:
    def __init__(self,mc):
        cmac_module_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "CMAC.py"
        m = imp.load_source('',cmac_module_location)
        self.QvalueCMAC = m.QvalueCMAC({'pos':(-1.2,0.6,8),'vel':(-0.07,0.07,9)},10,["neutral","forward","reverse"],Q0=1)
        self.start_of_episode = True
        self.s = None
        self.a = None
        self.old_qvalue = None
        self.mc = mc
        self.steps = 0
        self.best_steps = 1e6

    def one_step(self):
        self.steps += 1
        if self.start_of_episode:
            self.QvalueCMAC.reset_eligibility()
            self.s = {'pos':mc.pos,'vel':mc.velocity}
            self.a = self.QvalueCMAC.get_best_action_for_state(self.s)
            self.old_qvalue = self.QvalueCMAC.get_qvalue(self.s,self.a)
            self.start_of_episode = False

        self.QvalueCMAC.update_eligibilities(self.s,self.a)
        self.mc.move(self.a)
        if not mc.target_reached():
            r = -1
            s_ = {'pos':mc.pos,'vel':mc.velocity}
            a_ = self.QvalueCMAC.get_best_action_for_state(s_)
            new_qvalue = self.QvalueCMAC.get_qvalue(s_,a_)
            self.QvalueCMAC.update_weights(r,self.old_qvalue,new_qvalue) #update
            self.old_qvalue = self.QvalueCMAC.get_qvalue(s_,a_)

            self.a = a_
            self.s = s_#copy.deepcopy(s_)
            return False
        else:
            self.QvalueCMAC.update_weights(0,self.old_qvalue,0) #update
            self.start_of_episode = True
            self.best_steps = min(self.steps,self.best_steps)
            self.mc.__init__()
            return True

class MountainCarQLearn:
    def __init__(self,mc):
        cmac_module_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "CMAC.py"
        m = imp.load_source('',cmac_module_location)
        self.QvalueCMAC = m.QvalueCMAC({'pos':(-1.2,0.6,8),'vel':(-0.07,0.07,9)},10,["neutral","forward","reverse"],Q0=0)
        self.start_of_episode = True
        self.s = None
        self.a = None
        self.old_qvalue = None
        self.mc = mc
        self.steps = 0
        self.best_steps = 1e6
        self.epsilon = 0.05

    def _with_probability(self,one_minus_epsilon):
        if (np.random.rand() < one_minus_epsilon): return True
        return False



    def one_step(self):
        self.steps += 1
        epsilon = self.epsilon / float(self.steps)
        if self.start_of_episode:
            self.QvalueCMAC.reset_eligibility()
            self.s = {'pos':mc.pos,'vel':mc.velocity}
            self.a = self.QvalueCMAC.get_best_action_for_state(self.s)
            self.old_qvalue = self.QvalueCMAC.get_qvalue(self.s,self.a)
            self.start_of_episode = False

        self.QvalueCMAC.update_eligibilities(self.s,self.a)
        self.mc.move(self.a)
        if not mc.target_reached():
            r = -1
            s_ = {'pos':mc.pos,'vel':mc.velocity}

            best_action =self.QvalueCMAC.get_best_action_for_state(s_)
            if self._with_probability(1-epsilon):
                a_ = best_action
            else:
                a_ = np.random.choice(["neutral","forward","reverse"])

            if a_ == best_action:
                new_qvalue = self.QvalueCMAC.get_qvalue(s_,a_)
                self.QvalueCMAC.update_weights(r,self.old_qvalue,new_qvalue) #update
                self.old_qvalue = self.QvalueCMAC.get_qvalue(s_,a_)
            else:
                self.QvalueCMAC.reset_eligibility()
            self.a = a_
            self.s = s_
            return False
        else:
            self.QvalueCMAC.update_weights(0,self.old_qvalue,0) #update
            self.start_of_episode = True
            self.best_steps = min(self.steps,self.best_steps)
            self.mc.__init__()
            return True


def position(mca):
    x = [mca.mc.pos,mca.mc.pos]
    y = [np.sin(3*mca.mc.pos),np.sin(3*mca.mc.pos)+0.3]
    return (x, y)


def animation_init():
    """initialize animation"""
    x = np.linspace(-1.2,0.6)
    y = np.sin(3*x)
    line.set_data([x], [y])
    line.set_marker(None)
    time_text.set_text('')
    return line, time_text

def learn(mca):
    target_reached = mca.one_step()
    if target_reached:
        print "target reached after " + str(mca.steps) + " steps"
        mca.steps = 0


def animate(i,mca):
    """perform animation step"""
    learn(mca)
    line.set_data(position(mca))
    time_text.set_text('#steps = %d' % mca.steps)
    return line, time_text



if __name__ == "__main__":
    mc = MountainCar()
    mca = MountainCarSarsaLearn(mc)

    while mca.best_steps > (120 * int(plot == True)):
        learn(mca)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ani = animation.FuncAnimation(fig, animate, frames=None, interval=0, blit=True, init_func=animation_init,fargs=[mca])
    plt.show()

