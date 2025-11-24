import numpy as np
from bayspec.model.local import *
from bayspec import DataUnit, Data, Infer, Plot


savepath = './quickstart'

nai = DataUnit(
    src='./ME/me.src', 
    bkg='./ME/me.bkg', 
    rsp='./ME/me.rsp', 
    notc=[8, 900], 
    stat='pgstat', 
    rebn={'min_sigma': 2, 'max_bin': 10})

bgo = DataUnit(
    src='./HE/he.src', 
    bkg='./HE/he.bkg', 
    rsp='./HE/he.rsp', 
    notc=[300, 38000], 
    stat='pgstat', 
    rebn={'min_sigma': 2, 'max_bin': 10})

data = Data([('nai', nai), ('bgo', bgo)])
data.save(savepath)
print('<data information>')
print(data)

model = cpl()
model.save(savepath)
print('<model information>')
print(model)

infer = Infer([(data, model)])
infer.save(savepath)
print('<infer information>')
print(infer)

post = infer.multinest(nlive=400, resume=True, savepath=savepath)
post.save(savepath)
print('<post information>')
print(post)

fig = Plot.infer(post, style='CE', ploter='plotly')
fig.save(f'{savepath}/ctsspec')

fig = Plot.infer(post, style='CE', ploter='matplotlib')
fig.save(f'{savepath}/ctsspec')

fig = Plot.infer(post, style='NE', ploter='plotly')
fig.save(f'{savepath}/phtspec')

fig = Plot.infer(post, style='NE', ploter='matplotlib')
fig.save(f'{savepath}/phtspec')

fig = Plot.infer(post, style='Fv', ploter='plotly')
fig.save(f'{savepath}/flxspec')

fig = Plot.infer(post, style='Fv', ploter='matplotlib')
fig.save(f'{savepath}/flxspec')

fig = Plot.infer(post, style='vFv', ploter='plotly')
fig.save(f'{savepath}/ergspec')

fig = Plot.infer(post, style='vFv', ploter='matplotlib')
fig.save(f'{savepath}/ergspec')

fig = Plot.post_corner(post, ploter='plotly')
fig.save(f'{savepath}/corner')

fig = Plot.post_corner(post, ploter='getdist')
fig.save(f'{savepath}/corner')

earr = np.logspace(1, 3, 100)
modelplot = Plot.model(ploter='plotly', style='vFv', post=True)
modelplot.add_model(model, E=earr)
fig = modelplot.get_fig()
fig.save(f'{savepath}/model')

earr = np.logspace(1, 3, 100)
modelplot = Plot.model(ploter='matplotlib', style='vFv', post=True)
modelplot.add_model(model, E=earr)
fig = modelplot.get_fig()
fig.save(f'{savepath}/model')
