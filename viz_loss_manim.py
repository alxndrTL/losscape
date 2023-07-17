from manim import *
import random
import math
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import h5py
from scipy.interpolate import RectBivariateSpline
 
BG_COLOR = "#1B1818"
P_WHITE = "#F7F5EE"
P_BLUE = '#277da1'
P_YELLOW = '#ffb700'
P_GREEN = '#a1c181'
P_RED = '#d00000'
P_BLUE = "#277da1"
 
class M1(ThreeDScene):
    def construct(self):
        resolution_fa = 16
        self.set_camera_orientation(phi=np.pi/4, theta=-np.pi/2, zoom=0.5)
        axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(-2, 2, 0.5))
 
        def param_surface(u, v):
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return z
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-5, 5],
            u_range=[-5, 5],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)
 
class M0(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
 
        x_label = axes.get_x_axis_label(Tex("x"))
        y_label = axes.get_y_axis_label(Tex("y")).shift(UP * 1.8)
 
        # 3D variant of the Dot() object
        dot = Dot3D(color=RED)
 
        self.add(axes)
        self.add(dot)
 
class M2(ThreeDScene):
    def construct(self):
        resolution_fa = 8 #16 est bien
        self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) # vue du dessus
        #self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) #np.pi/4 -np.pi/2
        axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(-2, 2, 0.5))
 
        def param_surface(u, v):
            x = u
            y = v
            z = np.cos(x) * np.sin(y)
            return z
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-5, 5],
            u_range=[-5, 5],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)
 
        """
        dots = VGroup()
        for i in range(0, 100):
            x = 0 + i/10*0.1
            y = np.pi/2 + i/10*0.1
 
            dot = Dot3D(color=P_BLUE, radius=0.01)
            dot.move_to(axes.c2p(x, y, param_surface(x, y)))
            dots.add(dot)
        self.add(dots)
        """
 
        points = [(0, 0, 0), (1, 1, 1), (2, 1, 4), (0, 2, 4)]
        path = VMobject()
        path.set_points_smoothly([np.array(point) for point in points])
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)
 
        #self.begin_ambient_camera_rotation(rate=0.15)
        #self.wait(5)



#saddle
class M3(ThreeDScene):
    def construct(self):
        resolution_fa = 8 #16 est bien
        self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) # vue du dessus
        #self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) #np.pi/4 -np.pi/2
        axes = ThreeDAxes(x_range=(-1, 1, 1), y_range=(-1, 1, 1), z_range=(-1, 1, 0.5))
 
        def param_surface(u, v):
            x = u
            y = v
            z = np.square(x) - np.square(y)
            return z
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-1, 1],
            u_range=[-1, 1],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)
 
        points = [(0.0, 0.10000000149011612, -0.010000000707805157),
            (0.0, 0.12000000476837158, -0.014400000683963299),
            (0.0, 0.14400000870227814, -0.020736003294587135),
            (0.0, 0.1728000044822693, -0.029859840869903564),
            (0.0, 0.20735999941825867, -0.042998168617486954),
            (0.0, 0.24883200228214264, -0.06191736459732056),
            (0.0, 0.29859840869903564, -0.08916100859642029),
            (0.0, 0.3583180904388428, -0.12839184701442719),
            (0.0, 0.42998170852661133, -0.18488426506519318),
            (0.0, 0.5159780383110046, -0.26623332500457764),
            (0.0, 0.6191736459732056, -0.38337600231170654),
            (0.0, 0.7430083751678467, -0.5520614385604858),
            (0.0, 0.8916100263595581, -0.7949684262275696),
            (0.0, 1.069931983947754, -1.144754409790039),
            (0.0, 1.2839183807373047, -1.6484464406967163),
            (0.0, 1.5407021045684814, -2.373763084411621),
            (0.0, 1.8488425016403198, -3.4182186126708984),
            (0.0, 2.218611001968384, -4.922235012054443),
            (0.0, 2.6623332500457764, -7.088018417358398),
            (0.0, 3.1947999000549316, -10.206746101379395)]
        path = VMobject()
        path.set_points_smoothly([np.array(point) for point in points])
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)
 
        #self.begin_ambient_camera_rotation(rate=0.15)
        #self.wait(5)
 
class M4(ThreeDScene):
    def construct(self):
        resolution_fa = 8 #16 est bien
        self.set_camera_orientation(phi=70*DEGREES, theta=300*DEGREES, zoom=0.5) # vue du dessus
        
        #self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) #np.pi/4 -np.pi/2
        axes = ThreeDAxes(x_range=(-2, 2, 1), y_range=(-2, 2, 1), z_range=(-2, 1, 0.5))

        gaussian = lambda t, mu, sigma: 1/(sigma * (np.sqrt(2*np.pi))) * torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

        def f(x, y):
            z = -gaussian(x, 1., 0.2) * gaussian(y, -0.5, 0.2)
            z -= gaussian(x, -1., 0.2) * gaussian(y, 0.5, 0.2)
            z -= gaussian(x, -0.5, 0.2) * gaussian(y, -0.5, 0.2)
            return z

        gaussianv_viz = lambda t, mu, sigma: 1/(sigma * (np.sqrt(2*np.pi))) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

        def f_viz(x, y):
            z = -gaussianv_viz(x, 1., 0.2) * gaussianv_viz(y, -0.5, 0.2)
            z -= gaussianv_viz(x, -1., 0.2) * gaussianv_viz(y, 0.5, 0.2)
            z -= gaussianv_viz(x, -0.5, 0.2) * gaussianv_viz(y, -0.5, 0.2)
            return max(z, -2)
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, f_viz(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, 2],
            u_range=[-2, 2],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)

        #dot = Dot3D(color=RED)
        #dot.move_to((0.16, 0.16, cost(0.16, 0.15)))
        #self.add(dot)

        params = torch.tensor([0., 0.], requires_grad=True)

        points = []
        for i in range(200):
            loss = f(params[0], params[1])

            params.grad = None

            loss.backward()

            points.append((params[0].item(), params[1].item(), loss.item()))
            
            params.data -= 0.01 * params.grad
        
        path = VMobject()
        path.set_points_smoothly([np.array(point) for point in points])
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)
        
 
        #self.begin_ambient_camera_rotation(rate=0.15)
        #self.wait(5)

#cosxsin(y)
class M5(ThreeDScene):
    def construct(self):
        resolution_fa = 8 #16 est bien
        self.set_camera_orientation(phi=70*DEGREES, theta=210*DEGREES, zoom=0.5) # vue du dessus
        
        #self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) #np.pi/4 -np.pi/2
        axes = ThreeDAxes(x_range=(-1, 3, 1), y_range=(-4.5, 1.5, 1), z_range=(-2, 1, 0.5))


        def f(x, y):
            return torch.cos(x)*torch.sin(y)

        def f_viz(x, y):
            return np.cos(x)*np.sin(y)
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, f_viz(u, v)),
            resolution=(resolution_fa, resolution_fa),
            u_range=[-1, 3],
            v_range=[-4.5, 1.5],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)

        #dot = Dot3D(color=RED)
        #dot.move_to(axes.c2p(2., -2., f_viz(2., -2.)))
        #self.add(dot)

        params = torch.tensor([2., -2.], requires_grad=True)

        points = []
        for i in range(1000):
            loss = f(params[0], params[1])

            params.grad = None

            loss.backward()

            point = axes.c2p(params[0].item(), params[1].item(), loss.item())
            points.append(point)
            
            params.data -= 0.01 * params.grad
        
        path = VMobject()
        path.set_points_smoothly(points)
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)
        
 
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(5)

# --------------------------------------------------------------------------------------------------------------

def generate_spiral_dataset(N, K):
    X = np.zeros((N*K, 2)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N) # rayon
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X, y

N = 100 # nombre de points par classe
K = 3 # nombre de classes
X, Y = generate_spiral_dataset(N, K)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y)

device = 'cpu'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, axis=1)

model = MLP()
model.to(device)

checkpoint = torch.load("last_model")
model.load_state_dict(checkpoint)

def eval_loss(model):
    correct = 0

    logits = model(X)

    loss = F.cross_entropy(logits, Y)

    pred = F.log_softmax(logits, dim=1).argmax(dim=1, keepdim=True)
    correct = pred.eq(Y.view_as(pred)).sum().item()
    
    return loss.item(), correct/len(X)

def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]

def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)

    return direction

def get_weights(model):
    return [p.data for p in model.parameters()]

def get_random_weights(weights):
    return [torch.randn(w.size()).to(device) for w in weights]

def normalize_direction(direction, weights):
    for d, w in zip(direction, weights): 
        d.mul_(w.norm() / (d.norm() + 1e-10)) 

def normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0) 
        normalize_direction(d, w)

rand_directions = create_random_directions(model)
init_weights = [p.data for p in model.parameters()]

def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0] # Direction vector present in the scale of weights
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)] #  αδ + βη
    
    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w + d # θ^* + αδ + βη

def loss_at(x, y):
    overwrite_weights(model, init_weights, rand_directions, np.array([x, y]))
    loss, acc = eval_loss(model)

    #print(x, y, loss)
    return loss


#test loss landscape NN
class N1(ThreeDScene):
    def construct(self):

        resolution_fa = 16#8 #16 est bien
        self.set_camera_orientation(phi=50*DEGREES, theta=210*DEGREES, zoom=0.5) # vue du dessus
        
        #self.set_camera_orientation(phi=45*DEGREES, theta=70*DEGREES, zoom=0.5) #np.pi/4 -np.pi/2
        axes = ThreeDAxes(x_range=(-1, 1, 1), y_range=(-1, 1, 1), z_range=(0, 1, 0.5))
 
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, loss_at(u, v)),
            resolution=(resolution_fa, resolution_fa),
            u_range=[-0.5, 0.5],
            v_range=[-0.5, 0.5],
            )
        surface_plane.set_style(fill_opacity=0.5)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes, surface_plane)

        #dot = Dot3D(color=RED)
        #dot.move_to(axes.c2p(2., -2., f_viz(2., -2.)))
        #self.add(dot)

        """
        params = torch.tensor([2., -2.], requires_grad=True)

        points = []
        for i in range(1000):
            loss = f(params[0], params[1])

            params.grad = None

            loss.backward()

            point = axes.c2p(params[0].item(), params[1].item(), loss.item())
            points.append(point)
            
            params.data -= 0.01 * params.grad
        
        path = VMobject()
        path.set_points_smoothly(points)
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)
        """
        
 
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(5)
