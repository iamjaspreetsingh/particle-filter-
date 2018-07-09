
# coding: utf-8

# In[153]:


import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from collections import namedtuple


# In[315]:



def func( eq, yk, st ):
    k = st.k;
    Ns = st.Ns;                             
    nx = np.size(st.particles,1);              

    
    wkm1 = st.w[: k-1];                    
    if k == 2:
        for i in range(1,Ns):
            st.particles[:,i,1] = st.gen_x0(); 
        wkm1 = np.matlib.repmat(1/Ns, Ns, 1); 

    xkm1 = st.particles[:,:,k-1];
    xk   = np.zeros((np.size(xkm1)));    
    wk   = np.zeros((np.size(wkm1)));    

    for i in range (1,Ns):
        xk[:,i] = eq(k, xkm1[:,i], st.gen_eq_noise());
        wk[i] = wkm1[i] * st.p_yk_given_xk(k, yk, xk[:,i]);

    wk = wk/sum(wk);   #./
    Neff = 1/sum(wk**2);# .^

    resample_percentage = 0.50;
    Nt = resample_percentage*Ns;
    if Neff < Nt: 
        [xk, wk] = resample(xk, wk);

    xhk = zeros(nx,1);
    for i in range [1:Ns]:
        xhk = xhk + wk(i)*xk[:,i];

    st.w[:,k] = wk;
    st.particles[:,:,k] = xk;
    return xhk, st


# In[316]:


def resample( xk, wk ):
    Ns = length(wk); 
    edges = min([0,np.cumsum(wk)],1);
    #edges(end) = 1;                 
    u1 = rand/Ns;
    [idx]=~np.digitize(u1[:1]/Ns[:1], edges);
    xk = xk[: idx];                   
    wk = np.matlib.repmat(1/Ns, 1, Ns);          
    return xk, wk,idx;


# In[317]:


nx = 1;
eq = lambda k, xkm1, uk : xkm1/2 + 25*xkm1/(1+xkm1**2) + 8*math.cos(1.2*k) + uk; 


# In[318]:


ny = 1;
obs = lambda k, xk, vk : xk**2/20 + vk;                  
nu = 1;      


# In[319]:


sigma_u = math.sqrt(10);
p_eq_noise   = lambda u : normpdf(u, 0, sigma_u);
gen_eq_noise = lambda u : np.random.normal(0, sigma_u);         


# In[320]:


nv = 1;                                           
sigma_v = math.sqrt(1);
p_obs_noise   = lambda v : np.random.normal(v, 0, sigma_v);
gen_obs_noise = lambda v : np.random.normal(0, sigma_v);         
gen_x0 = lambda x : normrnd(0, sqrt(10));               


# In[321]:


p_yk_given_xk = lambda k, yk, xk : p_obs_noise(yk - obs(k, xk, 0));
T = 100;


# In[322]:


x = np.zeros((nx,T));
y = np.zeros((ny,T));
u = np.zeros((nu,T));
v = np.zeros((nv,T));
xh0 = 0;                          


# In[323]:


u[:1] = 0;                               
v[:1] = gen_obs_noise(sigma_v);          
x[:1] = xh0;
y[:1] = obs(1, xh0, v[:1]);


# In[324]:


for k in range(2,T):
    u[:k] = gen_eq_noise(k);
    v[:k] = gen_obs_noise(k);              
    x[:k] = eq(k, x[:k-1], u[:k]);     
    y[:k] = obs(k, x[:k],   v[:k]);


# In[325]:


xh = np.zeros((nx, T)); xh[:1] = xh0;
yh = np.zeros((ny, T)); yh[:1] = obs(1, xh0, 0);


# In[326]:


xh = np.zeros((nx, T));
xh[:1] = xh0;
y = np.zeros((ny, T));
yh[:1] = obs(1, xh0, 0);


# In[327]:


st = namedtuple("st", "k"  "Ns"  "w"  "particles"  "gen_x0"  "p_yk_given_xk"  "gen_eq_noise");

st.k               = 1;                   
st.Ns              = 500;                 
st.w               = np.zeros((st.Ns, T));     
st.particles       = np.zeros((nx, st.Ns, T)); 
st.gen_x0          = gen_x0;              
st.p_yk_given_xk   = p_yk_given_xk;       
st.gen_eq_noise   = gen_eq_noise;  


# In[328]:


for k in range(2,T):
    st.k = k;
    [xh[:k], st] = func(eq, y[:k], st);    
    yh[:k] = obs(k, xh[:k], 0);
    


# In[285]:



rmse=math.sqrt(np.mean((y-yh)**2))
rmse


# In[239]:


#plt.plot([1:T],y,'r', 1:T,yh,'g');
#plt.show()
#plt.legend('observation','filtered observation');
#plt.title('Observation and Filtered Observation','FontSize',14);

