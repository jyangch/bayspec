import numpy as np
from bayspec.model.local import *
from bayspec.util.tools import json_dump
from bayspec import DataUnit, Data, Infer, Plot


savepath = './quickstart'

nai = DataUnit(
    src='./ME/me.src', 
    bkg='./ME/me.bkg', 
    rsp='./ME/me.rsp', 
    notc=[8, 900], 
    stat='pgstat', 
    grpg={'min_sigma': 3, 'max_bin': 10})

bgo = DataUnit(
    src='./HE/he.src', 
    bkg='./HE/he.bkg', 
    rsp='./HE/he.rsp', 
    notc=[300, 38000], 
    stat='pgstat', 
    grpg={'min_sigma': 3, 'max_bin': 10})

data = Data([('nai', nai), ('bgo', bgo)])
json_dump(data.info.data_list_dict, savepath + '/data.json')
print('<data information>')
print(data)

model = cpl()
json_dump(model.cfg_info.data_list_dict, savepath + '/model_cfg.json')
json_dump(model.par_info.data_list_dict, savepath + '/model_par.json')
print('<model information>')
print(model)

infer = Infer([(data, model)])
json_dump(infer.cfg_info.data_list_dict, savepath + '/infer_cfg.json')
json_dump(infer.par_info.data_list_dict, savepath + '/infer_par.json')
print('<infer information>')
print(infer)

post = infer.emcee(nstep=1000, resume=True, savepath='./quickstart')
json_dump(post.free_par_info.data_list_dict, savepath + '/post_free_par.json')
json_dump(post.stat_info.data_list_dict, savepath + '/post_stat.json')
json_dump(post.IC_info.data_list_dict, savepath + '/post_IC.json')
print('<post information>')
print(post)

fig = Plot.infer_ctsspec(post, style='CE', ploter='plotly', show=False)
fig.write_html(savepath + '/ctsspec.html')
json_dump(fig.to_dict(), savepath + '/ctsspec.json')

fig = Plot.infer_ctsspec(post, style='CE', ploter='matplotlib', show=False)
fig.savefig(savepath + '/ctsspec.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

fig = Plot.post_corner(post, ploter='plotly', show=False)
fig.write_html(savepath + '/corner.html')
json_dump(fig.to_dict(), savepath + '/corner.json')

fig = Plot.post_corner(post, ploter='matplotlib', show=False)
fig.savefig(savepath + '/corner.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

earr = np.logspace(np.log10(0.5), 3, 100)
modelplot = Plot.model(ploter='plotly', style='vFv', CI=True)
fig = modelplot.add_model(model, E=earr, show=False)
fig.write_html(savepath + '/model.html')
json_dump(fig.to_dict(), savepath + '/model.json')

earr = np.logspace(np.log10(0.5), 3, 100)
modelplot = Plot.model(ploter='matplotlib', style='vFv', CI=True)
fig = modelplot.add_model(model, E=earr, show=False)
fig.savefig(savepath + '/model.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
