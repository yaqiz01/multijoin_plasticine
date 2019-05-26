#!/opt/local/bin/python
import matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from cycler import cycler
import pandas as pd
from model import *

def getcolors(name,num):
    cmap = plt.cm.get_cmap(name,num)
    return [cmap(i) for i in range(num)]

def plot_join3():
    fig, (ax1, ax2) = plt.subplots(1,2)
    plot_join3_rbkt(fig, ax1)
    plot_join3_smbkt(fig, ax2)
    plot_path = 'images/join3.pdf'
    fig.set_size_inches(6,3)
    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', dpi=900)
    print('Generate {}'.format(plot_path))

def plot_join3_rbkt(fig=None, ax=None):
    sizes = np.linspace(100, 15000, 100, dtype=int)
    xs = []
    times = []
    bott3 = []
    for x in sizes:
        try:
            param = run_algo(algo=join3, Hh_rec=x, Hgh_rec=1, N=100000000, d=1000000)
            xs.append(x)
            times.append(param['min'])
            bott3.append(param['join_bottleneck'])
        except AssertionError as e:
            # print(e)
            pass
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    # ax.plot(xs, times)
    markers = {'comp':'o', 'stream_T':'s', 'load_S':'^', 'load_R':'x'}
    for b in set(bott3):
        mask = np.array(bott3) == b
        xdata = np.array(xs)[mask]
        ydata = np.array(times)[mask]
        # ax.plot(xdata, ydata, marker=markers[b], markevery=10, 
            # label='bottleneck={}'.format(b))
        ax.plot(xdata, ydata, markevery=10, label='bottleneck={}'.format(b))
    minx = xs[np.argmin(times)]
    ax.axvline(x = minx, color='black', linestyle='dashed')
    ax.text(minx+max(xs)*0.05, max(times)/2, '$|R_{{bkt}}|={}$'.format(minx))
    # ax.legend(loc=1, fontsize='x-small')
    ax.set_xlabel(r'$|R_{bkt}|=\frac{|R|}{H_3h_3}$')
    ax.set_ylabel('3-way runtime (min)')
    if nofig:
        plot_path = 'images/join3_rbkt.pdf'
        fig.set_size_inches(4,3)
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_join3_bkt(fig=None, ax=None):
    nofig = fig is None
    if nofig: _, ax = plt.subplots()

    d = 600000000
    N = 300*d
    npmu = 64
    # H_bkts = np.linspace(d/npmu/10, d / npmu, 5)
    H_bkts = np.linspace(3*(10**6),5*(10**6),3)
    g_bkts = np.logspace(2.4,6, 100)
    Y = H_bkts
    X = g_bkts
    ax.grid(True, linestyle='--')
    markers = {'comp':'o', 'stream_T':'s', 'load_S':'^', 'load_R':'x'}
    mc = {'comp':'orchid', 'stream_T':'beige', 'load_S':'goldenrod', 'load_R':'lavendar'}
    for ih, H_bkt in enumerate(H_bkts):
        xs, ys, bott, _ = get_join('g_bkt', g_bkts, 'min', algo=join3, d=d, N=N, npmu=npmu, H_bkt=H_bkt)
        ax.plot(xs, ys, label='H_bkt={}'.format(int(H_bkt)))
        for b in set(bott):
            mask = np.array(bott) == b
            xdata = np.array(xs)[mask]
            ydata = np.array(ys)[mask]
            label = r'{}'.format(b) if ih == len(H_bkts)-1 else None
            ax.plot(xdata, ydata, marker=markers[b], markevery=20, mfc=mc[b],
                    color='none', label=label)
    ax.set_xscale('log')

    ax.plot([-1],[-1], color='none',label='N={}\nd={}'.format(N,d))
    ax.legend(loc=1, fontsize='x-small')
    ax.set_xlabel(r'$g_{bkt}$')
    ax.set_ylabel(r'Self: 3-way Join Runtime (min)')
    ax.set_ylim(bottom=0)
    ax.set_xlim([min(X),max(X)])
    if nofig:
        plot_path = 'images/join3_bkt.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_join3_rbkt_best(fig=None, ax=None):
    sizes = np.linspace(100, 15000, 100, dtype=int)
    Ns = np.linspace(1000000, 250000000, 100, dtype=int)
    xs = []
    ys = []
    for x in Ns:
        rbkts = []
        times = []
        bs = []
        for rbkt in sizes:
            try:
                param = run_algo(algo=join3, Hh_rec=rbkt, Hgh_rec=1, N=x, d=x/100)
                rbkts.append(rbkt)
                times.append(param['min'])
            except AssertionError as e:
                # print(e)
                pass
        if len(rbkts) != 0:
            bestr = np.argmin(times)
            xs.append(x)
            ys.append(rbkts[bestr])
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    ax.plot(xs, ys, markevery=10)
    ax.set_xlabel('N')
    ax.set_ylabel(r'Best $|R_{bkt}|=\frac{|R|}{H_3h_3}$')
    if nofig:
        plot_path = 'images/join3_rbkt_best.pdf'
        fig.set_size_inches(4,3)
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_join3_smbkt(fig=None, ax=None):
    sizes = np.concatenate(([1], np.linspace(2, 100, 100, dtype=int)))
    times = []
    for x in sizes:
        param = run_algo(algo=join3, Hgh_rec=x, Hh_rec=4*1024)
        times.append(param['min'])
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    ax.plot(sizes, times)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$|S_{minibkt}|=\frac{|S|}{H_3g_3h_3}$')
    ax.set_ylabel('3-way runtime (min)')
    if nofig:
        plot_path = 'images/join3_smbkt.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_join2_bkt(fig=None, axs=None):
    nofig = fig is None
    if fig is None:
        fig, axs = plt.subplots(1,2)

    param = init_param(d=10**6)
    bkts = np.linspace(param['d']/param['npmu']/10, param['d']/param['npmu'], 20, dtype=int)
    for i, key in enumerate(['H_bkt', 'G_bkt']):
        ax = axs[i]
        xs = []
        timesRS = []
        timesTRS = []
        times = []
        bott1 = []
        bott2 = []
        for x in bkts:
            try:
                param = run_algo(algo=join2, d=param['d'], **{key:x})
                xs.append(x)
                timesRS.append(param['RS_join_lat'] * 1.0 / param['clock'] / 60)
                timesTRS.append(param['TRS_join_lat'] * 1.0 / param['clock'] / 60)
                times.append(param['min'])
                bott1.append(param['join_RS_bottleneck'])
                bott2.append(param['join_TRS_bottleneck'])
            except AssertionError as e:
                # traceback.print_exc()
                print(e)
                pass
        ax.grid(True, linestyle='--')
        markers = {'comp':'o', 'store_RS':'s', 'stream_RS':'^', 'load_R':'x'}
        mc = {'comp':'orchid', 'store_RS':'beige', 'stream_RS':'goldenrod', 'load_R':'lavendar'}
        ax.fill_between(xs, [0]*len(xs), times, label='partition', color='C1')
        ax.fill_between(xs, [0]*len(xs), np.array(timesRS)+np.array(timesTRS), 
                label=r'$RS\bowtie T$', color='C2')
        ax.fill_between(xs, [0]*len(xs), np.array(timesRS), label=r'$R\bowtie S$', color='C3')
        for b in set(bott2):
            mask = np.array(bott2) == b
            xdata = np.array(xs)[mask]
            ydata = np.array(timesTRS)[mask]
            ydata += np.array(timesRS)[mask]
            ax.plot(xdata, ydata, marker=markers[b], markevery=3, mfc=mc[b],
                    color='none', 
                    label=r'{}'.format(b))
        for b in set(bott1):
            mask = np.array(bott1) == b
            xdata = np.array(xs)[mask]
            ydata = np.array(timesRS)[mask]
            ax.plot(xdata, ydata, marker=markers[b], markevery=3, mfc=mc[b],
                    color='none', label=r'{}'.format(b))
        ax.plot([-1], [-1], color='none', label='d={}\nN={}'.format(param['d'],param['N']))
        handles, labels = ax.get_legend_handles_labels()
        if key == 'H_bkt':
            order = [0,1,3,4,5,2]
        else:
            order = [0,1,2,4,5,3]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=0, fontsize='x-small')
        if key == 'H_bkt':
            ax.set_xlabel(r'$H_{bkt}$')
        else:
            ax.set_xlabel(r'$G_{bkt}$')
        ax.set_ylim(bottom=0)
        ax.set_xlim([min(xs),max(xs)])
        ax.set_ylabel('Self: 2-way Runtime (min)')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))

    if nofig:
        plot_path = 'images/join2_bkt.pdf'
        fig.set_size_inches(6,3)
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_3vs2_runtime(fig=None, ax=None, Hh_rec=4*1024):
    Ns = np.linspace(1000000, 250000000, 100, dtype=int)
    xs3 = []
    times3 = []
    bott3 = []
    xs2 = []
    times2 = []
    bott2 = []
    # xs2s = []
    # times2s = []

    for x in Ns:
        x = long(x)
        d = x / 100
        try:
            param = run_algo(algo=join3, N=x, 
                Hh_rec=Hh_rec,
                d=d)
            xs3.append(x)
            times3.append(param['min'])
            bott3.append(param['join_bottleneck'])
        except AssertionError as e:
            # print(e)
            pass
        try:
            param = run_algo(algo=join2, N=x, d=d, 
                    H_bkt=lambda param: divup(param['R_size'], param['npmu'] * param['pmu_cap_word'] / 2)*8,
                    G_bkt=lambda param: divup(param['T_size'], param['npmu'] * param['pmu_cap_word'] / 2)*8
                )
            xs2.append(x)
            times2.append(param['min'])
        except AssertionError as e:
            # print(e)
            pass

    nofig = fig is None
    if nofig:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    markers = {'comp':'o', 'stream_T':'s'}
    for b in set(bott3):
        mask = np.array(bott3) == b
        xdata = np.array(xs3)[mask]
        ydata = np.array(times3)[mask]
        ax.plot(xdata, ydata, marker=markers[b], markevery=10, 
            label='3-way (bottleneck={})'.format(b))
    ax.plot(xs2, times2, label='2-way')
    # ax.plot(xs2s, times2s, label='2-way-slow')
    ax.legend(loc=0, fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel('3-way runtime')
    if nofig:
        plot_path = 'images/join3vs2.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def get_best_join3(xkey, xlist, ykey, **kws):
    xs = []
    ys = []
    for x in xlist:
        kws[xkey] = x
        param = init_param(**kws)
        H_bkts = np.linspace(max(1,param['d']/param['npmu']/100), param['d']/param['npmu'], 20)
        g_bkts = np.linspace(max(1,param['d']/10000), param['d'], 20)
        besty = None
        bestx = None
        for H_bkt in H_bkts:
            for g_bkt in g_bkts:
                try:
                    param = run_algo(g_bkt=g_bkt, H_bkt=H_bkt, **kws)
                    y = param[ykey]
                    if besty is None or y < besty:
                        besty = y
                        bestx = x
                except AssertionError as e:
                    # print(e)
                    pass
        if besty is not None:
            xs.append(bestx)
            ys.append(besty)
    return np.array(xs),np.array(ys),None,None

def get_join(xkey, xlist, ykey, **kws):
    xs = []
    ys = []
    bott1 = []
    bott2 = []
    for x in xlist:
        try:
            kws[xkey] = x
            param = run_algo(**kws)
            xs.append(x)
            ys.append(param[ykey])
            if kws['algo'] == join3:
                bott1.append(param['join_bottleneck'])
            elif kws['algo'] == join2:
                bott1.append(param['join_RS_bottleneck'])
                bott2.append(param['join_TRS_bottleneck'])
        except AssertionError as e:
            # print(e)
            pass
    return np.array(xs), np.array(ys), np.array(bott1), np.array(bott2)

# DSE
def get_join_speedup(xkey, xlist, ykey, **kws):
    xs3, ys3, _, _ = get_best_join3(xkey, xlist, ykey, algo=join3, **kws)
    # xs3, ys3, _, _ = get_join(xkey, xlist, ykey, algo=join3, **kws)
    xs2, ys2, _, _ = get_join(xkey, xlist, ykey, algo=join2, **kws)
    xs,m3,m2 = np.intersect1d(xs3,xs2,return_indices=True)
    ys = ys2[m2] * 1.0 / ys3[m3]
    return xs, ys

def plot_join_N(fig=None, ax=None, algo=join2):
    # Ns = np.linspace(1000000, 100000000, 100, dtype=int)
    Ns = np.logspace(5, 8, 100, dtype=int)
    nofig = fig is None
    if nofig:
        _, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for i, f in enumerate([300, 500, 1000]):
        xs, ys, bott, _ = get_join('N',Ns,'min',algo=algo,d= lambda param: param['N'] / f)
        ax.plot(xs, ys, label=r'$f=\frac{N}{d}$'+'={}'.format(f))
    ax.legend(loc=0, fontsize='x-small')
    ax.set_xlabel('N')
    if algo==join2:
        ax.set_ylabel('Self: 2-way Runtime')
    else:
        ax.set_ylabel('3-way runtime')
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,max(Ns)])
    if nofig:
        if algo==join2:
            plot_path = 'images/join2_N.pdf'
        else:
            plot_path = 'images/join3_N.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_3vs2_speedup_d(fig=None, ax=None):
    Ns = np.linspace(1000000, 100000000*8, 20, dtype=int)
    nofig = fig is None
    if nofig:
        _, ax = plt.subplots()
    ax.grid(True, linestyle='--', axis='y')
    c = ['C1','C2','C3']
    for i, f in enumerate([300, 500, 1000]):
        rs_not_fit = [notfitindram('RS', init_param(N=N, d=lambda param: param['N']/f)) for N in Ns]
        notfit = Ns[np.where(rs_not_fit)[0][0]]
        xs, ys = get_join_speedup('N',Ns,'min',d= lambda param: param['N']/f)
        # xs, ys,_,_ = get_join('N',Ns,'min',algo=join3,d= lambda param: param['N'] / f)
        ax.plot(xs, ys, c[i],label=r'$\frac{N}{d}$'+'={}'.format(f))
        ax.axvline(x=notfit, color=c[i], linestyle='dashed')
        # ax.text(notfit-1.6e8,max(ys)*0.8,r'$|R\bowtie S|>2TB$')

    ax.plot(Ns, [1]*len(Ns), linestyle='--', color='black')


    # ax.plot([-1],[-1], color='none',label='49GB/s (DDR3)')
    # ax.plot([-1],[-1], color='none',label='700MB/s (SSD)')
    ax.legend(loc=1, fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel(r'Self: 3 vs. 2-way Speedup')
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,max(Ns)])
    if nofig:
        plot_path = 'images/join3vs2_speedup_d.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_3vs2_speedup_bw(fig=None, ax=None):
    Ns = np.linspace(1000000, 100000000*8, 20, dtype=int)
    nofig = fig is None
    if nofig:
        _, ax = plt.subplots()

    ax.grid(True, linestyle='--')
    for bw in [10, 49, 200]:
        xs, ys = get_join_speedup('N',Ns,'cycle',d= lambda param: param['N'] / 300, dram_bw_GBs=bw)
        ax.plot(xs, ys, label='{}GB/s (DDR3)'.format(bw))
    ax.plot(Ns, [1]*len(Ns), linestyle='--', color='black')

    # ax.plot([-1],[-1], color='none',label=r'$f=\frac{N}{d}$'+'={}'.format(f))
    ax.legend(loc=1, fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel(r'Self: 3 vs. 2-way Speedup')
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,max(Ns)])
    if nofig:
        plot_path = 'images/join3vs2_speedup_bw.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_3vs2_perfwatt(fig=None, ax=None):
    Ns = np.linspace(1000000, 10000000, 100, dtype=int)
    nofig = fig is None
    if nofig:
        _, ax = plt.subplots()

    ax.grid(True, linestyle='--')
    for f in [10, 100, 200, 300]:
        xs, ys = get_join_speedup('N',Ns,'total_energy',d= lambda param: param['N'] / f)
        ax.plot(xs, ys, label=r'$f=\frac{N}{d}$'+'={}'.format(f))
    ax.plot(Ns, [1]*len(Ns), linestyle='--', color='gray')

    ax.legend(loc=1, fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel('Perf/Watt Improvement')
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,max(Ns)])
    if nofig:
        plot_path = 'images/join3vs2_perfwatt.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))


def plot_3vs2():
    fig, axs = plt.subplots(3,3, gridspec_kw = {'hspace':0.4, 'wspace':0.4})
    faxs = np.array(axs).reshape(1,9)[0]
    i=0
    ax = faxs[i]; i+=1; plot_join_N(fig, ax, algo=join2)
    ax = faxs[i]; i+=2; plot_join2_bkt(fig, [ax, faxs[i-1]])
    ax = faxs[i]; i+=1; plot_join_N(fig, ax, algo=join3)
    ax = faxs[i]; i+=1; plot_join3_rbkt(fig, ax)
    ax = faxs[i]; i+=1; plot_join3_rbkt_best(fig, ax)
    ax = faxs[i]; i+=1; plot_join3_smbkt(fig, ax)
    ax = faxs[i]; i+=1; plot_3vs2_speedup_d(fig, ax)
    ax = faxs[i]; i+=1; plot_3vs2_speedup_bw(fig, ax)
    for r,row in enumerate(axs):
        for c,ax in enumerate(row):
            ax.set_title('({})'.format(chr(97+r*len(row)+c)),color='red',loc='left', pad=-145)
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
            # if c != 0:
                # ax.set_ylabel('')
    plot_path = 'images/join3vs2.pdf'
    fig.set_size_inches(7,7)
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.98)
    plt.savefig(plot_path, format='pdf', dpi=900)
    print('Generate {}'.format(plot_path))

def plot_all():
    fig, axs = plt.subplots(3,3, gridspec_kw = {'hspace':0.4, 'wspace':0.4})
    faxs = np.array(axs).reshape(1,9)[0]
    i=0
    ax = faxs[i]; i+=2; plot_join2_bkt(fig, [ax, faxs[i-1]])
    ax = faxs[i]; i+=1; plot_cpu_comp(fig, ax)
    ax = faxs[i]; i+=1; plot_join3_bkt(fig, ax)
    ax = faxs[i]; i+=1; plot_3vs2_speedup_d(fig, ax)
    ax = faxs[i]; i+=1; plot_3vs2_speedup_bw(fig, ax)
    ax = faxs[i]; i+=1; plot_sbsjoin3_h_bkt(fig, ax)
    ax = faxs[i]; i+=1; plot_sbsjoin_speedup_d(fig, ax)
    ax = faxs[i]; i+=1; plot_sbsjoin_speedup_K(fig, ax)
    for r,row in enumerate(axs):
        for c,ax in enumerate(row):
            ax.set_title('({})'.format(chr(97+r*len(row)+c)),color='red',loc='left', pad=-140)
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
        # if c != 0:
            # ax.set_ylabel('')
    plot_path = 'images/all.pdf'
    fig.set_size_inches(7,7)
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.98)
    plt.savefig(plot_path, format='pdf', dpi=900)
    print('Generate {}'.format(plot_path))

def get_sbs(xkey, xlist, ykey, **kws):
    xs = []
    ys = []
    hs = [1,2,4,16,32,64]
    for x in xlist:
        try:
            kws[xkey] = x
            if kws['algo'] == sbsjoin3:
                yys = []
                for h_bkt in hs:
                    try:
                        param = run_algo(h_bkt=h_bkt, **kws)
                        yys.append(param[ykey])
                    except AssertionError as e:
                        pass
                assert(len(yys)>0)
                y = min(yys)
            else:
                param = run_algo(**kws)
                y = param[ykey]
            xs.append(x)
            ys.append(y)
        except AssertionError as e:
            pass
    return xs, ys

def plot_sbsjoin3_h_bkt(fig=None, ax=None):
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    hs = range(1,65,1)
    K=1000
    N=10**8
    for d in np.linspace(64,64*3,3):
        ys = []
        xs = []
        for x in hs:
            try:
                param = run_algo(algo=sbsjoin3, h_bkt=x, K=K, N=N, d=d)
                xs.append(x)
                ys.append(param['sec'])
            except AssertionError as e:
                pass
        ax.plot(xs, ys, label='d={}'.format(d))
    ax.plot([-1], [-1], color='none', label='K={}\nN={}'.format(param['K'],param['N']))
    ax.legend(fontsize='x-small')
    ax.set_xlabel('$h_{bkt}$')
    ax.set_ylabel('Star: 3-way Runtime (sec)')
    ax.set_ylim(bottom=0, top=max(ys)*1.5)
    ax.set_xlim(0, max(hs))
    if nofig:
        plt.tight_layout()
        plot_path = 'images/sbs3_h.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_d(fig=None, ax=None):
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    bws = [10, 49, 400]
    algos = [sbsjoin2, sbsjoin3]
    ds = np.linspace(10, 2000, 100, dtype=int)

    ax.grid(True, linestyle='--')
    cyc = cycler(linestyle=['-','--']) * cycler(color=getcolors("Paired", 8)[1:1+len(bws)]) 
    ax.set_prop_cycle(cyc)
    name = {sbsjoin2:'(binary)', sbsjoin3:'(3-way)'}
    for algo in algos:
        for bw in bws:
            times = []
            xs = []
            for x in ds:
                try:
                    param = run_algo(algo=algo, dram_bw_GBs=bw, d=x)
                    xs.append(x)
                    times.append(param['sec'])
                except AssertionError as e:
                    # print(e)
                    pass
            ax.plot(xs, times, markevery=10, label='DDR3:{}GB/s {}'.format(bw, name[algo]))
    ax.legend(fontsize='x-small')
    ax.set_xlabel('d')
    ax.set_ylabel('Runtime (sec)')
    ax.set_ylim(bottom=0)
    if nofig:
        plot_path = 'images/sbs_d.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_K(fig=None, ax=None):
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    bws = [10, 49, 500]
    algos = [sbsjoin2, sbsjoin3]
    Ks = np.linspace(100, 10000, 100, dtype=int)

    ax.grid(True, linestyle='--')
    cyc = cycler(marker=['^','o'][0:len(algos)]) * cycler(color=getcolors("Paired", 8)[1:1+len(bws)]) 
    ax.set_prop_cycle(cyc)
    name = {sbsjoin2:'binary join', sbsjoin3:'3-way join'}
    for algo in algos:
        for bw in bws:
            xs, ys = get_sbs('K',Ks,'min',dram_bw_GBs=bw,algo=algo)
            ax.plot(xs, ys, markevery=10, label='DDR3:{}GB/s {}'.format(bw, name[algo]))
    ax.legend(fontsize='x-small')
    ax.set_xlabel('K')
    ax.set_ylabel('Runtime (min)')
    ax.set_ylim(bottom=0)
    if nofig:
        plot_path = 'images/sbs_K.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_N(fig=None, ax=None):
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ds = [100, 500, 1000]
    algos = [sbsjoin2, sbsjoin3]
    Ns = np.linspace(1000000, 100000000, 100, dtype=int)

    ax.grid(True, linestyle='--')
    cyc = cycler(linestyle=['-','--']) * cycler(color=getcolors("Paired", 8)[1:1+len(ds)]) 
    ax.set_prop_cycle(cyc)
    name = {sbsjoin2:'(binary)', sbsjoin3:'(3-way)'}
    for algo in algos:
        for d in ds:
            xs, ys = get_sbs('N',Ns,'min',d=d,algo=algo)
            ax.plot(xs, ys, markevery=10, label='d={} {}'.format(d, name[algo]))
    ax.legend(fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel('Runtime (min)')
    ax.set_ylim(bottom=0)
    if nofig:
        plot_path = 'images/sbs_N.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_bw(fig=None, ax=None, K=1000, d=1000):
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    bw = [10,30,50,100,500]
    algos = [sbsjoin2, sbsjoin3]

    ax.grid(True, linestyle='--')
    # cyc = cycler(linestyle=['-','--']) * cycler(color=getcolors("Paired", 8)[1:1+len(ds)]) 
    # ax.set_prop_cycle(cyc)
    name = {sbsjoin2:'binary', sbsjoin3:'3-way'}
    for algo in algos:
        xs, ys = get_sbs('dram_bw_GBs',bw,'min',algo=algo, N=100000000, K=K, d=d)
        ax.plot(xs, ys, label='{}'.format(name[algo]))
    ax.set_ylim(bottom=0)
    ax.plot([-1],[-1], color='none', label='K={} d={}'.format(K,d))
    ax.legend(fontsize='x-small', loc='center right')
    ax.set_xlabel('bw (GB/s)')
    ax.set_ylabel('Runtime (min)')
    if nofig:
        plot_path = 'images/sbs_bw.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def get_sbs3vs2_speedup(xkey, xlist, ykey, **kws):
    xs = []
    ys = []
    hs = [1,2,4,16,32,64]
    for x in xlist:
        try:
            join2kws = kws.copy()
            join3kws = kws.copy()
            join2kws[xkey] = x
            join3kws[xkey] = x
            y3s = []
            for h_bkt in hs:
                try:
                    param = run_algo(algo=sbsjoin3, h_bkt=h_bkt, **join3kws)
                    y3s.append(param[ykey])
                except AssertionError as e:
                    pass
            assert(len(y3s)>0)
            y3 = min(y3s)
            param = run_algo(algo=sbsjoin2, **join2kws)
            y2 = param[ykey]
            xs.append(x)
            ys.append(y2 * 1.0 / y3)
        except AssertionError as e:
            pass
    return xs, ys

def plot_sbsjoin_speedup_d(fig=None, ax=None):
    bws = [20, 49, 100, 200, 500]
    ds = np.linspace(100, 1000, 100, dtype=int)
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for bw in bws:
        xs, ys = get_sbs3vs2_speedup('d',ds,'cycle', K=1000, N=100000000, dram_bw_GBs=bw)
        ax.plot(xs, ys, label='bw:{}GB/s'.format(bw))
    ax.plot(ds, [1]*len(ds), linestyle='--', color='black')
    ax.legend(fontsize='x-small')
    ax.set_xlim([min(ds), max(ds)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('d')
    ax.set_ylabel(r'Star: 3 vs. 2-way Speedup')
    if nofig:
        plot_path = 'images/sbs_speedup_d.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_speedup_K(fig=None, ax=None):
    bws = [30, 49, 100, 200, 400]
    Ks = np.linspace(100,2000,100, dtype=int)
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for bw in bws:
        xs, ys = get_sbs3vs2_speedup('K',Ks,'cycle', dram_bw_GBs=bw, N=100000000, 
                d=lambda param: param['K']/4)
        ax.plot(xs, ys, label='bw:{}GB/s'.format(bw))
    ax.plot(Ks, [1]*len(Ks), linestyle='--', color='black')
    ax.legend(fontsize='x-small')
    ax.set_xlim([min(Ks), max(Ks)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('K')
    ax.set_ylabel(r'Star: 3 vs. 2-way Speedup')
    if nofig:
        plot_path = 'images/sbs_speedup_K.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_perfwatt_d(fig=None, ax=None):
    Ns = [10000,1000000,10000000]
    ds = np.linspace(100, 1000, 100, dtype=int)
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for n in Ns:
        xs, ys = get_sbs3vs2_speedup('d',ds,'total_energy', K=1000, N=n)
        ax.plot(xs, ys, label='N={}'.format(n))
    ax.plot(ds, [1]*len(ds), linestyle='--', color='black')
    ax.legend(fontsize='x-small')
    ax.set_xlim([min(ds), max(ds)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('d')
    ax.set_ylabel('Perf/Watt Improvement')
    if nofig:
        plot_path = 'images/sbs_perfwatt_d.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin_speedup_N(fig=None, ax=None):
    Ns = np.linspace(1000000, 100000000, 100, dtype=int)
    ds = np.linspace(70, 2000, 5, dtype=int)
    nofig = fig is None
    if fig is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for d in ds:
        xs = []
        speedups = []
        for x in Ns:
            try:
                param = run_algo(algo=sbsjoin2, N=x, K=3000, d=d)
                e2 = param['cycle']
                param = run_algo(algo=sbsjoin3, N=x, K=3000, d=d)
                e3 = param['cycle']
                speedup = e2 * 1.0 / e3
                xs.append(x)
                speedups.append(speedup)
            except AssertionError as e:
                # print(e)
                # traceback.print_exc()
                pass
        ax.plot(xs, speedups, markevery=10, label='#Unique (d)'.format(d))
    ax.legend(fontsize='x-small')
    ax.set_xlabel('N')
    ax.set_ylabel(r'$Speedup_{\frac{3way}{binary}}$')
    if nofig:
        plot_path = 'images/sbs_speedup_N.pdf'
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def plot_sbsjoin():
    fig, axs = plt.subplots(2,3)
    faxs = np.array(axs).reshape(1,6)[0]
    i=0
    ax = faxs[i]; i+=1; plot_sbsjoin_bw(fig, ax, K=100, d=100)
    ax = faxs[i]; i+=1; plot_sbsjoin_bw(fig, ax, K=1000, d=1000)
    ax = faxs[i]; i+=1; plot_sbsjoin_bw(fig, ax, K=100000, d=100000)
    ax = faxs[i]; i+=1; plot_sbsjoin3_h_bkt(fig, ax)
    # ax = faxs[i]; i+=1; plot_sbsjoin_d(fig, ax)
    ax = faxs[i]; i+=1; plot_sbsjoin_speedup_d(fig, ax)
    ax = faxs[i]; i+=1; plot_sbsjoin_speedup_K(fig, ax)
    # plot_sbsjoin_perfwatt_d(fig, ax4)
    for r,row in enumerate(axs):
        for c,ax in enumerate(row):
            ax.set_title('({})'.format(chr(97+r*len(row)+c)),color='red',loc='left', pad=-155)
            # for tick in ax.get_yticklabels():
                # tick.set_rotation(90)
    plot_path = 'images/sbs.pdf'
    fig.set_size_inches(8,5)
    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', dpi=900)
    print('Generate {}'.format(plot_path))

def run_algo(**kvs):
    param = init_param(**kvs)
    param['algo'](param)
    derive_stat(param)
    return param

def plot_cpu_comp(fig=None, ax=None):
    df = pd.read_csv(
        'data/cpu_psql_cnt.csv', 
        header=0, # row for header file
        encoding="utf-8-sig",
    )
    
    data = []
    for index, row in df.iterrows():
        param = run_algo(algo=join2, d=int(row.d), N=int(row.N), dram_capacity_TB=0.251)
        param['sec']
        data.append(param['sec'])

    df['accel_time'] = pd.Series(data)
    df['speedup'] = df.cpu_time / df.accel_time
    df['dpct'] = df.d / df.N * 100

    nofig = fig is None
    if nofig: _, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    for dpct in df.dpct.unique():
        tab = df[df.dpct==dpct]
        tab = tab.sort_values('N')
        Ns = tab['N'].tolist()
        speedup = tab['speedup'].tolist()
        ax.plot(Ns, speedup, label='d%={}'.format(dpct))
    ax.legend(loc=0, fontsize='x-small')
    ax.set_xlabel(r'N')
    ax.set_ylabel('Speedup over CPU\n on Binary Joins (self)', multialignment='center')
    ax.set_xlim(min(Ns), max(Ns))
    if nofig:
        plot_path = 'images/cpu.pdf'
        plt.savefig(plot_path, format='pdf', dpi=900)
        print('Generate {}'.format(plot_path))

def main():
    # param = run_algo(algo=join2, d=800, N=1000)
    # param = run_algo(algo=join3)
    # param = run_algo(algo=join3, Hh_rec=1000, d=100000000/1000,N=100000000)
    # param = run_algo(algo=sbsjoin2)
    # param = run_algo(algo=sbsjoin3)
    # for k in param:
        # print(k + ":" + str(param[k]))

    # plot_join_N(algo=join3)
    # plot_join3_rbkt()
    # plot_join3_rbkt_best()
    # plot_join3_smbkt()
    # plot_join_N(algo=join2)
    # plot_3vs2_runtime()
    # plot_3vs2_speedup_d()
    # plot_3vs2_speedup_bw()
    # plot_3vs2_perfwatt()
    # plot_sbsjoin_d()
    # plot_sbsjoin_K()
    # plot_sbsjoin_N()
    # plot_sbsjoin_bw()
    # plot_sbsjoin3_h_bkt()
    # plot_sbsjoin_speedup_d()
    # plot_sbsjoin_speedup_N()
    # plot_sbsjoin_speedup_K()
    # plot_sbsjoin_perfwatt_d()
    # plot_3vs2()
    # plot_sbsjoin()
    # plot_cpu_comp()

    # plot_join3_bkt()
    # plot_join2_bkt()
    # plot_3vs2_speedup_d()
    plot_all()

if __name__ == "__main__":
    main()
