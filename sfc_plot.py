import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
from pyart.graph import cm
from datetime import datetime, timedelta
import sys
import numpy as np
from metpy.plots import ctables
import matplotlib.colors as colors
import pyart
import cStringIO
from PIL import Image

file2do=sys.argv[1]
print(file2do)

mpl.rc('font', family='monospace')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def landmarks():
    landmark_dict = {'C':(-64.2123,-31.3154),
                    'M':(-68.7987,-32.8278),
                    'E':(-64.26149,-27.79511),
                    'Y':(-64.7545,-32.1062),
                    'S':(-70.6693,-33.4489)}
    return landmark_dict

landmark_dict=landmarks()

cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.2, 0.36, 0.36),
                 (0.35, 0.86, 0.86),
                 (0.5,0.71,0.71),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.2,0.46, 0.46),
                   (0.35, 0.46, 0.46),
                   (0.5,0.24,0.24),
                   (1.0, 1.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.2,0.99, 0.99),
                  (0.35, 0.96, 0.96),
                  (0.5,0.24,0.24),
                  (1.0, 0.5, 0.0))}

N=56
rb_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,N)

#Color4:  30   0 0 0  0
#Color4: -20   255 255 255 255
#Color:  -40   80 86 98 188 251 255
#Color:  -50   0 13 178 0 0 87
#Color:  -60  21 233 0 0 107 14
#Color:  -75  213 20 0 99 4 0
#Color:  -90  82 79 72 255 255 0
#Color:  -110  77 79 92 255 255 255



cdict = {'red': ((0.0, 0.0, 1.0),
                 (0.2, 0.36, 0.36),
                 (0.35, 0.86, 0.86),
                 (0.5,0.71,0.71),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.2,0.46, 0.46),
                   (0.35, 0.46, 0.46),
                   (0.5,0.24,0.24),
                   (1.0, 1.0, 0.0)),
         'blue': ((0.0, 0.0, 1.0),
                  (0.2,0.99, 0.99),
                  (0.35, 0.96, 0.96),
                  (0.5,0.24,0.24),
                  (1.0, 0.5, 0.0))}


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    http://blogs.candoerz.com/question/161551/invert-colormap-in-matplotlib.aspx
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

rb_cmap_r = reverse_colourmap(rb_cmap)

#new_cmap = truncate_colormap(cmap, 0.2, 0.8)

wv_norm, wv_cmap = ctables.registry.get_with_steps('WVCIMSS', 0, 1)

wv_cmap = truncate_colormap(wv_cmap, 0.43, 1.0)

wv_cmap = reverse_colourmap(wv_cmap)

ir_norm, ir_cmap = ctables.registry.get_with_steps('ir_tpc', 0, 1)

ir_cmap = truncate_colormap(ir_cmap, 0.43, 1.0)

ir_cmap = reverse_colourmap(ir_cmap)

def findlevkey(inkeys):
    for key in inkeys:
        if key.startswith('lv'):
            return key

values=np.array([-110.,-90.,-75.,-60.,-50.,-40.,-35,-20.,-19.,30.])
norms=(values-np.min(values))/(np.max(values)-np.min(values))
rs=np.array([255,158,139,192,228,79,153,10,255,0])/255.
gs=np.array([255,84,13,95,225,162,224,78,255,0])/255.
bs=np.array([255,174,183,40,44,79,227,177,255,0])/255.

cdict = {'red': zip(norms,rs,rs),
         'green': zip(norms,gs,gs),
         'blue': zip(norms,bs,bs)}
N=255
rammb_cm=mpl.colors.LinearSegmentedColormap('rammb',cdict,N)

def make_sfc_map(filename=None,
                  level=None,
                  levtxt=None,
                  cfld=None,
                  cscl=None,
                  coff=None,
                  colormap=None,
                  cmin=None,
                  cmax=None,
                  cmaskblo=None,
                  cbunits=None,
                  nc=None,
                  z1fld=None,
                  z1levs=None,
                  z1off=None,
                  z1scl=None,
                  z2fld=None,
                  z2levs=None,
                  z2off=None,
                  z2scl=None,
                  ufld=None,
                  vfld=None,
                  titletext=None,
                  filepat=None):



    wrf=xr.open_dataset(filename,engine='pynio')

    run=filename.split('/')[-1]
    init=datetime.strptime(run.split('_')[0],'%Y%m%d%H')
    fhr=run.split('.')[1]
    valid=init+timedelta(hours=int(fhr))

#    levtxt='Surface'

#    level=float(level)*100.
    terr=wrf['HGT_P0_L1_GLC0'].values
#    hgt=wrf['HGT_P0_L100_GLC0'].values

    lon=wrf['gridlon_0'].values
    lat=wrf['gridlat_0'].values
    if len(np.shape(wrf[ufld].values)) == 2:
        ug=wrf[ufld].values
        vg=wrf[vfld].values
    else:
        levkey=findlevkey(wrf[ufld].coords.keys())
        dic={levkey: level}
        ug=wrf[ufld].sel(**dic).values
        vg=wrf[vfld].sel(**dic).values
    if cfld=='SPD':
        c=cscl*np.sqrt(ug**2+vg**2)+coff
    else:
        if len(np.shape(wrf[cfld].values)) == 2:
            c=cscl*wrf[cfld].values+coff
        else:
            levkey=findlevkey(wrf[cfld].coords.keys())
            dic={levkey: level}
            c=cscl*wrf[cfld].isel(**dic).values+coff
    if cmaskblo is not 'None':
        c=np.ma.masked_less_equal(c,cmaskblo)
    if z1fld is not 'None':
        if len(np.shape(wrf[z1fld].values)) == 2:
            z1=(z1scl*wrf[z1fld].values)+z1off
        else:
            levkey=findlevkey(wrf[z1fld].coords.keys())
            dic={levkey: level}
            z1=z1scl*wrf[z1fld].isel(**dic).values+z1off
    if z2fld is not 'None':
        z2=(z2scl*wrf[z2fld].values)+z2off
    fields=titletext
    u=ug*np.cos(wrf['gridrot_0'])-vg*np.sin(wrf['gridrot_0'])
    v=ug*np.sin(wrf['gridrot_0'])+vg*np.cos(wrf['gridrot_0'])

#    c=np.ma.masked_where(terr > hgt,c)
#    u=np.ma.masked_where(terr > hgt,u)
#    v=np.ma.masked_where(terr > hgt,v)
#    z1=np.ma.masked_where(terr > hgt,z1)
#    z2=np.ma.masked_where(terr > hgt,z2)
#    mask=ones_like(terr)
#    mask=np.ma.masked_where(terr < hgt+50,mask)

    m=Basemap(projection='lcc',width=3000*550,height=3000*375,
             resolution='i',lat_1=-32.8,lat_2=-32.8,lat_0=-32.8,lon_0=-67.0)
    x,y=m(lon,lat)
    N=19.

    font0 = FontProperties()
    font0.set_family('monospace')
    my_dpi=100
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    m.drawcoastlines()
    m.drawcountries(linewidth=1.0)
    m.drawstates(color=(0.5,0.5,0.5),linewidth=0.5)
    #m.drawparallels(np.arange(-80.,81.,1.))
    #m.drawmeridians(np.arange(-180.,181.,1.))
    C=m.pcolormesh(x,y,c,vmin=cmin,vmax=cmax,ax=ax,cmap=colormap)
    plt.title('Initialized '+init.isoformat()+' UTC\n'
              'F'+fhr+' Valid '+valid.isoformat()+' UTC',loc='right',fontdict={'family': 'monospace'})#plt.colorbar(orientation='horizontal',shrink=0.5,pad=0.0)
    plt.title('University of Illinois 3 km WRF Forecast\n'+
              fields,loc='left',fontdict={'family': 'sans-serif'})#plt.colorbar(orientation='horizontal',shrink=0.5,pad=0.0)
    for key in landmark_dict.keys():
        kx,ky=m(landmark_dict[key][0],landmark_dict[key][1])
        plt.text(kx,ky,key,fontsize=8,fontweight='light',
                        ha='center',va='center',color='b')
    if z1fld is not 'None':
        CS = m.contour(x,y,z1,z1levs,colors='k',lw=2,ax=ax)
        cl=plt.clabel(CS, fontsize=9, inline=1, fmt='%1.0f',fontproperties=font0)
    if z2fld is not 'None':
        CS2 = m.contour(x,y,z2,z2levs,colors='g',lw=2)
        for cl in CS2.collections:
            cl.set_dashes([(0, (.5, .5))])
        cl2=plt.clabel(CS2, fontsize=9, inline=1, fmt='%1.0f',fontproperties=font0,ax=ax)

    print(level)
    print(np.min(c))
    print(np.max(c))
    Urot, Vrot = m.rotate_vector(u,v,lon,lat)
    m.barbs(x[::20,::20],y[::20,::20],Urot[::20,::20],Vrot[::20,::20],
            barb_increments=dict(half=2.5, full=5., flag=25),
           length=5,flagcolor='none',barbcolor='k',lw=0.5,ax=ax)
    m.contour(x,y,terr,[500.,1500.],colors=('blue','red'),alpha=0.5)
    cax = fig.add_axes([0.2, 0.12, 0.4, 0.02])
    cb=plt.colorbar(C, cax=cax, orientation='horizontal')
    cb.set_label(cbunits, labelpad=-10, x=1.1)
    ram = cStringIO.StringIO()
    plt.savefig(ram, format='png',dpi=my_dpi, bbox_inches='tight')
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB')
    im2.save( run+'_'+filepat+'_'+levtxt+'_'+fhr+'.png' , format='PNG')

filepat='windmslp'
params={'Surface':{'crange':[0,40],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              cfld='SPD',
              levtxt='',
              cscl=1.0,
              coff=0.0,
              colormap=rb_cmap,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='m/s',
              nc=N,
              z1fld='PRMSL_P0_L101_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='10 m isotachs, MSLP',
              filepat=filepat)

filepat='rh'
params={'Surface':{'crange':[0,100],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              levtxt='',
              cfld='RH_P0_L103_GLC0',
              cscl=1.0,
              coff=0.0,
              colormap=mpl.cm.Greens,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='%',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='2 m RH, 10 m winds',
              filepat=filepat)

filepat='dwpt'
params={'Surface':{'crange':[-20,25],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              levtxt='',
              cfld='DPT_P0_L103_GLC0',
              cscl=1.0,
              coff=-273.15,
              colormap=rb_cmap_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='$^O$C',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='2 m dewpoint, 10 m winds',
              filepat=filepat)

filepat='pot'
params={'Surface':{'crange':[275,350],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              levtxt='',
              cfld='POT_P0_L1_GLC0',
              cscl=1.0,
              coff=0.0,
              colormap=rb_cmap_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='K',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='2 m potential temperature, 10 m winds',
              filepat=filepat)

filepat='tmp'
params={'Surface':{'crange':[-20,40],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              levtxt='',
              cfld='TMP_P0_L103_GLC0',
              cscl=1.0,
              coff=-273.15,
              colormap=rb_cmap_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='$^O$C',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='2 m temperature, 10 m winds',
              filepat=filepat)

filepat='capecin'
params={'0':{'type':'Boundary Layer','crange':[0,4000.],'z1levs':np.arange(-1000.,0.,100.),'z2levs':np.arange(-100.,40.,2)},
       '1':{'type':'Mixed Layer','crange':[0,4000.],'z1levs':np.arange(-1000.,0.,100.),'z2levs':np.arange(-100.,40.,2)},
       '2':{'type':'Most Unstable Layer','crange':[0,4000.],'z1levs':np.arange(-1000.,0.,100.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=int(i),
              cfld='CAPE_P0_2L108_GLC0',
              levtxt=params[i]['type'].replace (" ", "_"),
              cscl=1.0,
              coff=0.,
              colormap=rb_cmap,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='J/kg',
              nc=N,
              z1fld='CIN_P0_2L108_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=1.0,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext=params[i]['type']+' CAPE, CIN, 10 m winds',
              filepat=filepat)

filepat='sfcapecin'
params={'0':{'type':'Surface-based','crange':[0,4000.],'z1levs':np.arange(-1000.,0.,100.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=params[i]['type'],
              cfld='CAPE_P0_L1_GLC0',
              levtxt='Surface',
              cscl=1.0,
              coff=0.,
              colormap=rb_cmap,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='J/kg',
              nc=N,
              z1fld='CIN_P0_L1_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=1.0,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext=params[i]['type']+' CAPE, CIN, 10 m winds',
              filepat=filepat)

filepat='prec'
params={'Surface':{'crange':[0,100],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              cfld='APCP_P8_L1_GLC0_acc',
              levtxt='',
              cscl=1.0,
              coff=0.0,
              cmaskblo=0.0,
              colormap=cm.LangRainbow12,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='mm',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='Total precipitation',
              filepat=filepat)

filepat='preciph2o'
params={'Surface':{'crange':[20,60],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              cfld='PWAT_P0_L200_GLC0',
              levtxt='',
              cscl=1.0,
              coff=0.0,
              cmaskblo=0.0,
              colormap=cm.LangRainbow12,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='mm',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='Precipitable water',
              filepat=filepat)

filepat='ch3'
params={'Ch3':{'crange':[-100,0],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              cfld='SBT123_P0_L8_GLC0',
              levtxt='',
              cscl=1.0,
              coff=-273.15,
              colormap=wv_cmap,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='$^O$C',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='Simulated water vapor image',
              filepat=filepat)

filepat='ch4'
params={'Ch4':{'crange':[-100,50],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=i,
              cfld='SBT124_P0_L8_GLC0',
              levtxt='',
              cscl=1.0,
              coff=-273.15,
              colormap=rammb_cm,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='$^O$C',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='Simulated infrared image',
              filepat=filepat)


filepat='shr'
params={'0':{'type':'0-1 km','crange':[0,80],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)},
        '1':{'type':'0-6 km','crange':[0,80],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=int(i),
              levtxt=i,
              cfld='SPD',
              cscl=1.0,
              coff=0.0,
              cmaskblo=0.0,
              colormap=rb_cmap,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='m$^2$/s$^2$',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='VUCSH_P0_2L103_GLC0',
              vfld='VVCSH_P0_2L103_GLC0',
              titletext=params[i]['type']+' Shear, Shear Vectors',
              filepat=filepat)


filepat='helic'
params={'0':{'type':'0-1 km','crange':[0,500],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)},
        '1':{'type':'0-3 km','crange':[0,500],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=int(i),
              levtxt=params[i]['type'].replace (" ", "_"),
              cfld='HLCY_P0_2L103_GLC0',
              cscl=-1.0,
              coff=0.0,
              cmaskblo=0.0,
              colormap=cm.LangRainbow12,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='m$^2$/s$^2$',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext=params[i]['type']+' Helicity, Winds at ',
              filepat=filepat)

filepat='dbz1km'
params={'0':{'crange':[0,70],'z1levs':np.arange(980.,1040.,4.),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_sfc_map(filename=file2do,
              level=int(i),
              cfld='REFD_P0_L103_GLC0',
              levtxt='',
              cscl=1.0,
              coff=0.0,
              cmaskblo=0.0,
              colormap=cm.NWSRef,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='dBZe',
              nc=N,
              z1fld='None',
              z1levs=params[i]['z1levs'],
              z1scl=0.01,
              z1off=0.0,
              z2fld='None',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L103_GLC0',
              vfld='VGRD_P0_L103_GLC0',
              titletext='Radar Reflectivity - 1 km AGL',
              filepat=filepat)
