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


def make_plev_map(filename=None,
                  level=None,
                  cfld=None,
                  cscl=None,
                  coff=None,
                  colormap=None,
                  cmin=None,
                  cmax=None,
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


    level=float(level)*100.
    terr=wrf['HGT_P0_L1_GLC0'].values
    hgt=wrf['HGT_P0_L100_GLC0'].sel(lv_ISBL0=level,method='nearest').values

    lon=wrf['gridlon_0'].values
    lat=wrf['gridlat_0'].values
    ug=wrf[ufld].sel(lv_ISBL0=level,method='nearest').values
    vg=wrf[vfld].sel(lv_ISBL0=level,method='nearest').values
    u=ug*np.cos(wrf['gridrot_0'])-vg*np.sin(wrf['gridrot_0'])
    v=ug*np.sin(wrf['gridrot_0'])+vg*np.cos(wrf['gridrot_0'])

    if cfld=='SPD':
        c=cscl*np.sqrt(u**2+v**2)+coff
    elif cfld=='AVO':
        f=2.*7.29e-5*np.sin(lat* np.pi / 180.)
        dudx,dudy=np.gradient(u,3000.,edge_order=2)
        dvdx,dvdy=np.gradient(v,3000.,edge_order=2)
        c=cscl*(dvdx-dudy+f)+coff
    else:
        c=cscl*wrf[cfld].sel(lv_ISBL0=level).values+coff
    z1=(z1scl*wrf[z1fld].sel(lv_ISBL0=level).values)+z1off
    z2=(z2scl*wrf[z2fld].sel(lv_ISBL0=level).values)+z2off
    fields=titletext+str(int(level/100.))+' hPa'

    c=np.ma.masked_where(terr > hgt,c)
    u=np.ma.masked_where(terr > hgt,u)
    v=np.ma.masked_where(terr > hgt,v)
    z1=np.ma.masked_where(terr > hgt,z1)
    z2=np.ma.masked_where(terr > hgt,z2)
    mask=np.ones_like(terr)
    mask=np.ma.masked_where(terr < hgt+50,mask)

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
    CS = m.contour(x,y,z1,z1levs,colors='k',lw=2,ax=ax)
    cl=plt.clabel(CS, fontsize=9, inline=1, fmt='%1.0f',fontproperties=font0)
    CS2 = m.contour(x,y,z2,z2levs,colors='g',lw=2)
    for cl in CS2.collections:
        cl.set_dashes([(0, (2.0, 2.0))])
    print(level)
    print(np.min(c))
    print(np.max(c))
    cl2=plt.clabel(CS2, fontsize=9, inline=1, fmt='%1.0f',fontproperties=font0,ax=ax)
    Urot, Vrot = m.rotate_vector(u,v,lon,lat)
    m.barbs(x[::20,::20],y[::20,::20],Urot[::20,::20],Vrot[::20,::20],
            barb_increments=dict(half=2.5, full=5., flag=25),
           length=5,flagcolor='none',barbcolor='k',lw=0.5,ax=ax)
    m.contour(x,y,terr,[500.,1500.],colors=('blue','red'),alpha=0.5)
    m.pcolormesh(x,y,mask,cmap=cm.Gray5,vmin=0.,vmax=1.)
    cax = fig.add_axes([0.2, 0.12, 0.4, 0.02])
    cb=plt.colorbar(C, cax=cax, orientation='horizontal')
    cb.set_label(cbunits, labelpad=-10, x=1.1)
    ram = cStringIO.StringIO()
    plt.savefig(ram, format='png',dpi=my_dpi, bbox_inches='tight')
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB')
    im2.save( run+'_'+filepat+'_'+str(int(level/100.))+'_'+fhr+'.png' , format='PNG')

filepat='avohgttmp'
params={'200':{'crange':[-40,40],'z1levs':np.arange(1200,1400,3),'z2levs':np.arange(-100.,40.,2)},
        '250':{'crange':[-40,40],'z1levs':np.arange(900,1200,3),'z2levs':np.arange(-100.,40.,2)},
        '300':{'crange':[-40,40],'z1levs':np.arange(800,1000,3),'z2levs':np.arange(-100.,40.,2)},
        '500':{'crange':[-40,40],'z1levs':np.arange(400,600,3),'z2levs':np.arange(-100.,40.,2)},
        '700':{'crange':[-40,40],'z1levs':np.arange(200,400,3),'z2levs':np.arange(-100.,40.,2)},
        '850':{'crange':[-40,40],'z1levs':np.arange(100,200,1),'z2levs':np.arange(-100.,40.,2)},
        '925':{'crange':[-40,40],'z1levs':np.arange(0,100,1),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_plev_map(filename=file2do,
              level=i,
              cfld='AVO',
              cscl=-1e5,
              coff=0.0,
              colormap=mpl.cm.RdBu_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='-10$^{-5}$ m$^2$/s$^2$',
              nc=N,
              z1fld='HGT_P0_L100_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.1,
              z1off=0.0,
              z2fld='TMP_P0_L100_GLC0',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L100_GLC0',
              vfld='VGRD_P0_L100_GLC0',
              titletext='Abs. Vort., $\Phi$, Temperature at ',
              filepat=filepat)

filepat='windhgttmp'
params={'200':{'crange':[20,100],'z1levs':np.arange(1200,1400,3),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '250':{'crange':[20,100],'z1levs':np.arange(900,1200,3),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '300':{'crange':[20,100],'z1levs':np.arange(800,1000,3),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '500':{'crange':[20,100],'z1levs':np.arange(400,600,3),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '700':{'crange':[10,60],'z1levs':np.arange(200,400,3),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '850':{'crange':[10,60],'z1levs':np.arange(100,200,1),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'},
        '925':{'crange':[10,40],'z1levs':np.arange(0,100,1),'z2levs':np.arange(-100.,40.,2),'cmap':rb_cmap,'cbunits':'m/s'}}

for i in params:
    make_plev_map(filename=file2do,
              level=i,
              cfld='SPD',
              cscl=1.0,
              coff=0.0,
              colormap=params[i]['cmap'],
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits=params[i]['cbunits'],
              nc=N,
              z1fld='HGT_P0_L100_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.1,
              z1off=0.0,
              z2fld='TMP_P0_L100_GLC0',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L100_GLC0',
              vfld='VGRD_P0_L100_GLC0',
              titletext='Isotachs, $\Phi$, Temperature at ',
              filepat=filepat)

filepat='rhhgttmp'
params={'200':{'crange':[0,100],'z1levs':np.arange(1200,1400,3),'z2levs':np.arange(-100.,40.,2)},
        '250':{'crange':[0,100],'z1levs':np.arange(900,1200,3),'z2levs':np.arange(-100.,40.,2)},
        '300':{'crange':[0,100],'z1levs':np.arange(800,1000,3),'z2levs':np.arange(-100.,40.,2)},
        '500':{'crange':[0,100],'z1levs':np.arange(400,600,3),'z2levs':np.arange(-100.,40.,2)},
        '700':{'crange':[0,100],'z1levs':np.arange(200,400,3),'z2levs':np.arange(-100.,40.,2)},
        '850':{'crange':[0,100],'z1levs':np.arange(100,200,1),'z2levs':np.arange(-100.,40.,2)},
        '925':{'crange':[0,100],'z1levs':np.arange(0,100,1),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_plev_map(filename=file2do,
              level=i,
              cfld='RH_P0_L100_GLC0',
              cscl=1.0,
              coff=0.0,
              colormap=mpl.cm.Greens,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='%',
              nc=N,
              z1fld='HGT_P0_L100_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.1,
              z1off=0.0,
              z2fld='TMP_P0_L100_GLC0',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L100_GLC0',
              vfld='VGRD_P0_L100_GLC0',
              titletext='RH, $\Phi$, Temperature at ',
              filepat=filepat)

filepat='dpthgttmp'
params={'200':{'crange':[-80,20],'z1levs':np.arange(1200,1400,3),'z2levs':np.arange(-100.,40.,2)},
        '250':{'crange':[-80,20],'z1levs':np.arange(900,1200,3),'z2levs':np.arange(-100.,40.,2)},
        '300':{'crange':[-80,20],'z1levs':np.arange(800,1000,3),'z2levs':np.arange(-100.,40.,2)},
        '500':{'crange':[-80,20],'z1levs':np.arange(400,600,3),'z2levs':np.arange(-100.,40.,2)},
        '700':{'crange':[-80,20],'z1levs':np.arange(200,400,3),'z2levs':np.arange(-100.,40.,2)},
        '850':{'crange':[-40,25],'z1levs':np.arange(100,200,1),'z2levs':np.arange(-100.,40.,2)},
        '925':{'crange':[-20,25],'z1levs':np.arange(0,100,1),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_plev_map(filename=file2do,
              level=i,
              cfld='DPT_P0_L100_GLC0',
              cscl=1.0,
              coff=-273.15,
              colormap=rb_cmap_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='$^O$C',
              nc=N,
              z1fld='HGT_P0_L100_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.1,
              z1off=0.0,
              z2fld='TMP_P0_L100_GLC0',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L100_GLC0',
              vfld='VGRD_P0_L100_GLC0',
              titletext='Dewpoint, $\Phi$, Temperature at ',
              filepat=filepat)

filepat='whgttmp'
params={'200':{'crange':[-2,2],'z1levs':np.arange(1200,1400,3),'z2levs':np.arange(-100.,40.,2)},
        '250':{'crange':[-2,2],'z1levs':np.arange(900,1200,3),'z2levs':np.arange(-100.,40.,2)},
        '300':{'crange':[-2,2],'z1levs':np.arange(800,1000,3),'z2levs':np.arange(-100.,40.,2)},
        '500':{'crange':[-2,2],'z1levs':np.arange(400,600,3),'z2levs':np.arange(-100.,40.,2)},
        '700':{'crange':[-2,2],'z1levs':np.arange(200,400,3),'z2levs':np.arange(-100.,40.,2)},
        '850':{'crange':[-2,2],'z1levs':np.arange(100,200,1),'z2levs':np.arange(-100.,40.,2)},
        '925':{'crange':[-2,2],'z1levs':np.arange(0,100,1),'z2levs':np.arange(-100.,40.,2)}}

for i in params:
    make_plev_map(filename=file2do,
              level=i,
              cfld='VVEL_P0_L100_GLC0',
              cscl=1.0,
              coff=0.0,
              colormap=mpl.cm.RdBu_r,
              cmin=params[i]['crange'][0],
              cmax=params[i]['crange'][1],
              cbunits='m/s',
              nc=N,
              z1fld='HGT_P0_L100_GLC0',
              z1levs=params[i]['z1levs'],
              z1scl=0.1,
              z1off=0.0,
              z2fld='TMP_P0_L100_GLC0',
              z2levs=params[i]['z2levs'],
              z2scl=1.0,
              z2off=-273.15,
              ufld='UGRD_P0_L100_GLC0',
              vfld='VGRD_P0_L100_GLC0',
              titletext='$w$, $\Phi$, Temperature at ',
              filepat=filepat)
