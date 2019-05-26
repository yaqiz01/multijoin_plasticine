from collections import OrderedDict
import types
import traceback
import math
import numpy as np

# round up division
def divup(a,b):
    if isinstance(a, float) or isinstance(b, float):
        return long(math.ceil(a / b))
    else:
        return (a + b - 1) / b 

def notfitindram(table, param):
    return param[table + '_size_TB'] > param['dram_capacity_TB']

def comp_dram_lat(loadsize, drambwfrac, param, tp='dram'):
    bw = param[tp+'_bw_wordpcycle'] * drambwfrac
    return comp_pipe_lat_trpt(loadsize, bw, param['dram_lat_cycle'])

# Compute pipeline latency if throughput is less than 1
# interval between which the next iteration can start in pipeline.
# interval = 1 / throughput
def comp_pipe_lat_int(iters, interval, latency):
    return (iters - 1) * interval + latency

# Compute pipeline latency if throughput is more than 1
def comp_pipe_lat_trpt(N, throughput, latency):
    return divup(N, throughput) - 1 + latency

# Compute latency ff different stages have different iterations and is fine-grain pipelined
def comp_streampipe_lat(name, stages, param):
    iters = []
    tot_lat = 0
    for stagename, N, throughput, latency in stages:
        it = divup(N,throughput) - 1
        param[stagename+'_iter'] = it
        iters.append(it)
        tot_lat += latency
    stage = np.argmax(iters)
    param[name+'_bottleneck'] = stages[stage][0]
    return max(iters) + latency

# Compute meta pipeline latency, where pipe_lats is a list of latencies of each stage.
# Coarse grain pipelinig interval is the max(pipe_lats)
def comp_metapipe_lat(name, iters, stages, param):
    stagenames = [stagename for stagename, lat in stages]
    pipe_lats = [lat for stagename, lat in stages]
    stage = np.argmax(pipe_lats)
    stagename = stagenames[stage]
    max_pipe = max(pipe_lats)
    for stagename, lat in stages:
        param[stagename+'_lat'] = lat
    if stagename+'_bottleneck' in param:
        param[name+'_bottleneck'] = param[stagename+'_bottleneck']
    else:
        param[name+'_bottleneck'] = stagename
    param[name+'_iter'] = iters
    return comp_pipe_lat_int(iters, max_pipe, sum(pipe_lats))

# Profiling for groupby. 
# Load data on-chip and compute groupby keys. Flatten keys into 1-D key. 
# Keep counters of total number of buckets in PMU and count the distribution.
# Discard the hash value after count.
def profile_groupby_vec(table, hashes, param):
    ctr_size = reduce(lambda a,b: a * b, [param[h + "_bkt"] for h in hashes])
    npmu = divup(ctr_size, param['pmu_cap_word'])

    lat = 0
    # Load data from off-chip to on-chip
    loadsize = param[table + "_size"]
    lat += comp_dram_lat(loadsize, 1, param, tp='dram' if param[table+'_size_TB'] <= 2 else 'ssd')
    # Use enouth pcu and pmu to match throughput of DRAM
    outer_par = divup(param['dram_bw_wordpcycle'], param['pcu_lane'])
    ops = sum([param[h+'op'] for h in hashes])
    flatten_ops = (len(hashes) - 1) # assume flatten each dimension cost a FMA
    ops += flatten_ops
    npcu = divup(ops, param['pcu_stage'])
    compute_lat = npcu * param['pcu_stage']
    compute_lat += (npcu - 1) * param['net_lat']
    compute_lat += 1 # accumulation latency. Assume accumulation happens inside PMU
    lat += compute_lat
    npcu *= outer_par
    npmu = npmu * outer_par
    active = divup(param["N"], (outer_par * param['pcu_lane']))

    accum(param, [table + "_profile_cycle", 'cycle'], lat)
    accum(param, [table + '_profile_pcu_active', 'pcu_active'], npcu * active)
    accum(param, [table + '_profile_pmu_active', 'pmu_active'], npmu * active)

def hash_groupby_novec(table, hashes, param):
    # number of counters needed to track flatten hash bkt counts
    ctr_size = reduce(lambda a,b: a * b, [param[h + "_bkt"] for h in hashes])
    npmu = divup(ctr_size, param['pmu_cap_word']) # No need to double buffer for accumulator

    # Op latency to compute all hashes
    ops = sum([param[h+'op'] for h in hashes])
    # Flatten hashes if have more than one
    flatten_ops = (len(hashes) - 1) # assume flatten each dimension cost a FMA
    ops += flatten_ops
    # Number of chained pcus to compute hashes
    npcu = divup(ops, param['pcu_stage'])
    compute_lat = npcu * param['pcu_stage']
    # communication latency between DRAM interface => pipelined PCUs => PMU
    compute_lat += npcu * param['net_lat']
    compute_lat += 1 # accumulation latency. Assume accumulation happens inside PMU

    # Not parallelizing accumulation, since this is a groupby reduce. 
    # Would need reduction that can sends worst case 16 elements across lanes 
    # to parallelize this accumulation.
    # So throughput bounded by compute instead of DRAM. One record per cycle
    nrec = param[table+'_size'] / param[table+'_col']
    lat = comp_pipe_lat_trpt(nrec, 1, param['dram_lat_cycle'] + compute_lat)

    active = nrec 
    pcu_active = npcu * active
    pmu_active = npmu * active
    return ctr_size, lat, pcu_active, pmu_active

# Profiling for groupby. 
# Load data on-chip and compute groupby keys. Flatten keys into 1-D key. 
# Keep counters of total number of buckets in PMU and count the distribution.
# Discard the hash value after count.
def profile_groupby_novec(table, hashes, param):
    ctr_size, lat, pcu_active, pmu_active = hash_groupby_novec(table, hashes, param)

    accum(param, [table + "_profile_cycle", 'cycle'], lat)
    accum(param, [table + '_profile_pcu_active'], pcu_active)
    accum(param, [table + '_profile_pmu_active'], pmu_active)

# Use counter from profiling, generate bucket DRAM offset for each bucket.
def layout_groupby_novec(table, hashes, param):
    # Assume counters stayed on-chip after reconfiguration
    ctr_size, hash_lat, hash_pcu_active, hash_pmu_active = hash_groupby_novec(table, hashes, param)

    # First compute dram offset. 
    # PCU reads ctr, use a local register for accumulator, and add to each counter as offset.
    # Use another PMU to store the offset. Cannot vectorize
    #      pipeline lat + read latency + PMU => PCU + PCU lat + PCU => PMU + write latency
    accum(param, [table+"_profile_cycle", 'cycle'], 
        comp_pipe_lat_int(ctr_size, 1, param['net_lat'] + param['pcu_stage'] + param['net_lat'] + 1))
    accum(param, [table+'_profile_pcu_active', 'pcu_active'], ctr_size)
    # Read ctr_pmu and write to ofst_pmu
    accum(param, [table+'_profile_pmu_active', 'pmu_active'], ctr_size) 

    # Next reload data, recompute hashes, and change layout of data using dense scatter
    # operation, i.e. for each record, compute the storing address, and we know the storing
    # address should be adjacent. Coalescing unit inside DRAM interface should be able to
    # condense them into fewer burst requests.
    # Again, since compute is not parallelized, the entire process is bounded by compute
    # not DRAM load or store.
    accum(param, [table+"_profile_cycle", 'cycle'], hash_lat + param['net_lat'] +
            param['dram_lat_cycle'])
    accum(param, [table+'_profile_pcu_active', 'pcu_active'], hash_pcu_active)
    accum(param, [table+'_profile_pmu_active', 'pmu_active'], hash_pmu_active) 

def groupby(table, hashes, param):
    # Profiling for groupby
    # profile_groupby_vec(table, hashes, param)
    profile_groupby_novec(table, hashes, param)
    # Change layout for groupby
    # layout_groupby_vec(table, hashes, param)
    layout_groupby_novec(table, hashes, param)

# three-way join of three equal size tables
def join3(param):

    # 1. Group by data with hash values
    groupby('R', ['H','h'], param)
    groupby('S', ['H','g'], param)
    groupby('T', ['g'], param)

    # Assign bw proportional to loaded data
    R_loaded = param['RH_size'] * 1.0
    S_loaded = param['SH_size'] * 1.0
    T_loaded = param['T_size'] * 1.0
    total_loaded = (R_loaded + S_loaded + T_loaded)
    set_param(param,R_bw=R_loaded/total_loaded) # fraction of bw allocated to loading R
    set_param(param,S_bw=S_loaded/total_loaded) # fraction of bw allocated to loading S
    set_param(param,T_bw=T_loaded/total_loaded) # fraction of bw allocated to loading T
    # 2. Compute join. 

    # 2.1 Load R partition. Assume reserve 10% DRAM bandwidth to stream in S and T.
    load_R_lat = comp_dram_lat(param['RH_size'], param['R_bw'], param) + param['net_lat'] + 1

    # 2.2 Stream in S and T and compute Join
    # 2.2.1 Use 8% bandwidth to stream S into PCUs, compute hash h and compare with assigned hv. 
    # Assume can configure a small portion of PCU to do the hash compute, and has much higher
    # thoughput than DRAM load
    load_S_lat = comp_dram_lat(param['SHhg_size'], param['S_bw'], param) + param['net_lat'] + param['hop'] + 1
    param['load_S_lat'] = load_S_lat

    # 2.2.2 Compute join. Use 0.02% bandwidth Stream T from DRAM.
    stages = []
    # 2.2.2.1 Use a portion of bandwidth stream in T
    stages.append((
        'stream_T', 
        param['Tg_size'],
        param['dram_bw_wordpcycle'] * param['T_bw'], # throughput
        param['dram_lat_cycle'] + param['gop'] # lat
    ))
    # 2.2.2.2 Compare
    
    # T/g * (S/Hgh * P * (R/Hh)/L + 1 - P)
    #=T/g * (S/Hgh * g/d * (R/Hh)/L + 1 - g/d)
    #=(T*S/Hgh*R/Hh / dL) + (T*S/Hgh*(d-g)/gd)
    param['comp_ST'] = divup(param['Tg_rec'] * param['SHh_rec'], param['g_bkt'])
    param['hit'] = divup(param['comp_ST'] * param['g_bkt'] * 1.0, param['d'])
    param['miss'] = param['comp_ST'] - param['hit']
    param['total_comp'] = param['hit'] * divup(param['RHh_rec'], param['pcu_lane'])
    param['total_comp'] += param['miss']
    stages.append((
        'comp', 
        param['total_comp'], # N
        1,
        param['dram_lat_cycle'] + param['net_lat'] + param['pcu_stage'] # latency for comparison
    ))

    comp_lat = comp_streampipe_lat('stream_T_comp',stages, param)

    # 2.2.1 and 2.2.2 are pipelined
    stages = [('load_S', load_S_lat), 
        ('stream_T_comp', comp_lat)]
    load_S_comp_lat = comp_metapipe_lat('load_S_comp', param['g_bkt'], stages, param)

    # 2.1 and 2.2 are pipelined
    stages = [('load_R', load_R_lat), 
            ('load_S_comp', load_S_comp_lat)]
    accum(param,['pass2_lat','cycle'], comp_metapipe_lat('join', param['H_bkt'], stages, param))

    # use load active >> store active for R buckets. So use load active as pmu_active
    active = param['H_bkt'] * param['g_bkt'] * param['h_bkt'] * param['total_comp']
    accum(param, ['join_pmu_active', 'pmu_active'], active)
    accum(param, ['join_pcu_active', 'pcu_active'], active)

# three-way join with on-chip binary join of three equal size tables
def join32(param):

    # 1. Group by data with hash values
    groupby('R', ['H','h'], param)
    groupby('S', ['H','g'], param)
    groupby('T', ['g'], param)

    # 2. Compute join. 

    # 2.1 Load R partition. Assume reserve 10% DRAM bandwidth to stream in S and T.
    load_lat = comp_dram_lat(param['R_col']*param['H_rec'], 0.9, param) + param['net_lat'] + 1
    param['load_lat'] = load_lat

    # 2.2 Stream in S and T and compute Join
    # 2.2.1 Stream S into PCUs, compute hash h and compare with assigned hv. 
    # Assume using 10% bandwidth. 
    # Assume can configure a small portion of PCU to do the hash compute, and has much higher
    # thoughput than DRAM load
    load_S_lat = comp_dram_lat(param['SHhg_size'], 0.1, param) + param['net_lat'] + param['hop'] + 1

    # 2.2.2 Compute join. Stream T from DRAM. Assume throughput bounded by compute
    # 2.2.2.1 Join S with T
    param['st_comp_iter'] = param['g_rec'] * divup(param['Hgh_rec'],param['pcu_lane'])
    st_comp_lat = comp_pipe_lat_int(
            param['st_comp_iter'],
            1,
            param['dram_lat_cycle'] + param['net_lat'] + 1 # latency for comparison
        )
    param['st_comp_lat'] = st_comp_lat
    # 2.2.2.2 Join ST with R
    param['st_rec'] = param['g_rec'] * param['Hgh_rec'] * param['g_bkt'] * 1.0 / param['d']
    # param['st_size'] = param['st_rec'] * param['ST_col']
    # assert(param['st_size'] < 1000) # Can be stored in register
    param['str_comp_iter'] = long(math.ceil(param['st_rec'] * divup(param['RHh_rec'],
        param['pcu_lane'])))
    str_comp_lat = comp_pipe_lat_int(
            param['str_comp_iter'],
            1,
            param['net_lat'] + 1 # latency for comparison
        )
    # 2.2.2.1 and 2.2.2.2 are pipelined
    comp_lat = comp_metapipe_lat('comp', 
            divup(param['h_bkt'], param['npcu']), [st_comp_lat, str_comp_lat], param)

    # 2.2.1 and 2.2.2 are pipelined
    join_lat = comp_metapipe_lat('load_S_comp', param['g_bkt'], [load_S_lat, comp_lat], param)
    param['join_lat'] = join_lat

    # 2.1 and 2.2 are pipelined
    accum(param,['pass2_lat','cycle'],
            comp_metapipe_lat('join', param['H_bkt'], [load_lat, join_lat]), param)

    # Since 2.2.2.1 and 2.2.2.2 are using the same PCU, use which ever takes longer number iters as
    # active for PCU
    accum(param, ['join_pcu_active', 'pcu_active'], param['H_bkt'] * param['g_bkt'] * param['h_bkt'] * \
            max(param['g_rec'] * divup(param['Hgh_rec'], param['pcu_lane']),
                divup(param['g_rec'] * param['Hgh_rec'] * param['f'], param['N']) * 
                    divup(param['RHh_rec'], param['pcu_lane'])
                ))
    # use load active >> store active for R buckets. So use load active as pmu_active
    accum(param, ['join_pmu_active', 'pmu_active'], param['H_bkt'] * param['g_bkt'] * param['h_bkt'] * \
            param['g_rec'] * divup(param['Hgh_rec'], param['pcu_lane']))

# Hash T1 and T2 with h, load a chunck of T1 and T2, loop throughput T1 and T2
def join2_binary_slow(param, T1, T2, T3, h):
    T12 = T1+T2
    # First join
    # 1. Preprocessing
    groupby(T1, [h], param)
    groupby(T2, [h], param)
    # 2. Compute
    # 2.1 Load T1 and T2
    data_size = divup(param[T1+'_rec'], param[h+'_bkt'])*param[T1+'_col']
    data_size += divup(param[T2+'_rec'], param[h+'_bkt'])*param[T2+'_col']
    param[T12+'_load_lat'] = comp_dram_lat(data_size, 1, param)
    # 2.2 Join and store
    # 2.2.1
    stages = []
    num_compare = divup(param[T1+'_rec'], param[h+'_bkt']) * divup(param[T2+'_rec'], param[h+'_bkt'])
    stages.append((
        'comp', 
        num_compare, # N
        param['pcu_lane'] * param['npcu'], # throughput
        # load lat + net lat + compute lat + net lat
        1 + param['net_lat'] + param['pcu_stage'] + param['net_lat'] # latency
    ))
    if T3 is not None:
        bw = param['ssd_bw_wordpcycle'] if notfitindram(T3, param) else param['dram_bw_wordpcycle']
        stages.append((
            'store', 
            divup(param[T3+'_size'], param[h+'_bkt']), # output size of each tile
            bw, # throughput
            param['dram_lat_cycle'] # lat
        ))
    param[T12+'_comp_lat'] = comp_streampipe_lat('join_'+T12, stages, param)

    active = param[h+'_bkt'] * divup(num_compare, param['pcu_lane'])
    accum(param, [T12 + '_join_lat', 'cycle'], comp_metapipe_lat(param[h+'_bkt'],[param[T12+'_load_lat'],param[T12+'_comp_lat']]))
    accum(param, [T12 + '_join_pmu_active', 'pmu_active'], param[h+'_bkt'] * divup(param[T1+'_rec'],param[h+'_bkt'])) # Read of S
    accum(param, [T12 + '_join_pmu_active', 'pmu_active'], param[h+'_bkt'] * active) # Read of R
    accum(param, [T12 + '_join_pcu_active', 'pcu_active'], param[h+'_bkt'] * active)

# binary join of three equal size tables
def join2_slow(param):
    join2_binary_slow(param, 'R', 'S', 'RS', 'H')
    join2_binary_slow(param, 'RS', 'T', None, 'G')

# Hash T1 and T2 with H, which fits onchip.
# Load T1 chunk, and then hash by h, groupby onchip
# Streaming T2, and compare on chip.

def join2_binary(param, T1, T2, T3, H, h):
    if param[T1+'_rec'] > param[T2+'_rec']:
        tmp = T1
        T1 = T2
        T2 = tmp
    T12 = T1+T2
    # 1. Preprocessing
    groupby(T1, [H,h], param)
    groupby(T2, [H,h], param)

    tp = 'dram' if param[T2+'_size_TB'] <= 2 else 'ssd'
    # 2. Compute
    # Assign bw proportional to loaded data
    T1_loaded = param[T1+H+'_size']*1.0
    T2_loaded = param[T2+H+'_size']*1.0
    total_loaded = (T1_loaded + T2_loaded)
    param[T1+'_bw']=T1_loaded/total_loaded # fraction of bw allocated to loading R
    param[T2+'_bw']=T2_loaded/total_loaded # fraction of bw allocated to loading S

    # 2.1 Load T1
    T1_loadlat = comp_dram_lat(T1_loaded, param[T1+'_bw'], param, tp=tp) + param[h+'op']
    # 2.2 Join
    stages = []

    # 2.2.1 Stream in T2
    stages.append((
        'stream_'+T2, 
        T2_loaded,
        param[tp+'_bw_wordpcycle'] * param[T2+'_bw'], # throughput
        param['dram_lat_cycle'] + param[h+'op'] # lat
    ))
    # 2.2.2 Onchip comparisin
    num_compare = divup(param[h+'_bkt'], param['npcu']) * \
            divup(param[T2+H+'_rec'], param[h+'_bkt']) * \
            divup(param[T1+H+h+'_rec'], param['pcu_lane'])
    stages.append((
        'comp', 
        num_compare, # N
        1,
        # load lat + net lat + compute lat + net lat
        1 + param['net_lat'] + param['pcu_stage'] + param['net_lat'] # latency
    ))
    if T3 is not None:
        bw = param['ssd_bw_wordpcycle'] if notfitindram(T3, param) else param['dram_bw_wordpcycle']
        stages.append((
            'store_'+T3, 
            divup(param[T3+'_size'], param[H+'_bkt']), # output size of each tile
            bw, # throughput
            param['dram_lat_cycle'] # lat
        ))
    load_T2_comp = comp_streampipe_lat('load_'+T2+'_comp', stages, param)

    active = divup(num_compare,param['pcu_lane'])
    stages = [('load_'+T1, T1_loadlat),
            ('load_'+T2+'_comp', load_T2_comp)
        ]
    accum(param, [T12 + '_join_lat', 'cycle'], comp_metapipe_lat('join_'+T12,param[H+'_bkt'],stages,param))
    accum(param, [T12 + '_join_pmu_active', 'pmu_active'], param[h+'_bkt'] * active) # Read of R
    accum(param, [T12 + '_join_pcu_active', 'pcu_active'], param[h+'_bkt'] * active)

def join2(param):
    join2_binary(param, 'R', 'S', 'RS', 'H', 'h')
    join2_binary(param, 'RS', 'T', None, 'G', 'g')

# Groupby of small table. Result of groupby will
# be stored in different PMU.
def onchip_groupby(table, hashes, npmu, param):
    lat = 0
    load_lat = comp_dram_lat(param[table+'_size'], 1, param)
    lat += load_lat
    lat += param['net_lat'] # load to pcu
    hashop = sum([param[h+'op'] for h in hashes])
    npcu = divup(hashop, param['pcu_stage'])
    lat += npcu * param['pcu_stage'] + (npcu - 1) * param['net_lat'] # latency to compute hash
    lat += param['net_lat'] + 1 # send to PMUs and store
    active = divup(param[table+'_size'], param['pcu_lane'])
    return lat, npcu*active, npmu*active

# binary join small T1 and big T2 and store to T3. If no store T3 is None
def sbsjoin2_binary(param,T1,T2,T3,h):
    # T1 is the small table. T2 is the big table
    if param[T1+'_rec'] > param[T2+'_rec']:
        temp = T1
        T1 = T2
        T2 = temp
    T12 = T1+T2
    # 1.1 groupby T1
    groupby_lat, groupby_pcuactive, groupby_pmuactive = onchip_groupby(T1, h, param[h+'_bkt'], param)
    accum(param, ['groupby_lat', 'join_lat_'+T12, 'cycle'], groupby_lat)
    accum(param, ['groupby_pcu_active', 'join_pcu_active_'+T12, 'pcu_active'], groupby_pcuactive)
    accum(param, ['groupby_pmu_active', 'join_pmu_active_'+T12, 'pmu_active'], groupby_pmuactive)
    # 1.2 join T1 with T2
    # Determine throughput bottleneck. 
    # If store to T3 and join result is big, will be store bandwidth bound.
    # else 
    #   if T1 is small, it will be load bound on streaming T2
    #   if T1 is large enough, it will be compute bound

    stages = []
    stages.append((
        'load',
        param[T2+'_size'], # N
        param['dram_bw_wordpcycle'], # throughput
        param['dram_lat_cycle'] # lat
    ))
    # Total amt of compare
    param['tot_comp_rec_'+T12] = divup(param[T1+'_rec'] * param[T2+'_rec'], param[h+'_bkt'])
    stages.append((
        'comp',
        param['tot_comp_rec_'+T12], # N
        param['npcu'] * param['pcu_lane'], # throughput
        # fringe => pcu + compare + pcu => fringe
        param['net_lat'] + param['pcu_stage'] + param['net_lat'] # lat
    ))
    if T3 is not None:
        bw = param['ssd_bw_wordpcycle'] if notfitindram(T3, param) else param['dram_bw_wordpcycle']
        stages.append((
            'store',
            param[T3+'_size'], # N
            bw, # throughput
            param['dram_lat_cycle'] # lat
        ))

    # Use this compute throughput to compute latency
    accum(param, ['join_onchip_lat_'+T12, 'join_lat_'+T12, 'cycle'], 
        comp_streampipe_lat(T12+'_join', stages, param)
    )
    accum(param, ['join_pcu_active_'+T12, 'pcu_active'], 
        divup(param['tot_comp_rec_'+T12], param['pcu_lane'])
    )
    accum(param, ['join_pmu_active_'+T12, 'pmu_active'], 
        divup(param['tot_comp_rec_'+T12], param['pcu_lane'])
    )

# binary join of big S and small R and T.
def sbsjoin2(param):
    set_param(param,h_bkt=param['npmu']) # number of buckets for h
    assert(param['h_bkt'] <= param['d']), 'h_bkt={} > d={}'.format(param['h_bkt'], param['d'])
    set_param(param,h_rec=divup(param['K'], param['h_bkt'])) # bucket size of R hashed by h
    assert(param['h_rec'] <= divup(param['pmu_cap_word'], param['R_col']))
    set_param(param,g_bkt=param['npmu']) # number of buckets for g 
    set_param(param,g_rec=divup(param['K'], param['g_bkt'])) # bucket size of T hashed by g
    assert(param['g_rec'] <= divup(param['pmu_cap_word'], param['T_col']))

    sbsjoin2_binary(param,'R','S','RS','h')
    sbsjoin2_binary(param,'T','RS',None,'g')

# three-way join of big S and small R and T.
def sbsjoin3(param):
    set_param(param,g_bkt=param['npmu'] / param['h_bkt']) # number of buckets for g
    set_param(param,h_rec=divup(param['K'], param['h_bkt'])) # bucket size of R hashed by h
    assert(param['h_rec']*param['R_col'] <= param['pmu_cap_word'])
    set_param(param,g_rec=divup(param['K'], param['g_bkt'])) # bucket size of T hashed by g
    assert(param['g_rec']*param['T_col'] <= param['pmu_cap_word'])

    param['used_cu'] = param['h_bkt'] * param['g_bkt']
    # 1. groupby R and T
    groupby_lat, groupby_pcuactive, groupby_pmuactive = onchip_groupby('R', 'h', param['used_cu'],
            param)
    accum(param, ['groupby_lat', 'cycle'], groupby_lat)
    accum(param, ['groupby_pcu_active', 'pcu_active'], groupby_pcuactive)
    accum(param, ['groupby_pmu_active', 'pmu_active'], groupby_pmuactive)
    groupby_lat, groupby_pcuactive, groupby_pmuactive = onchip_groupby('T', 'g', param['used_cu'],
            param)
    accum(param, ['groupby_lat', 'cycle'], groupby_lat)
    accum(param, ['groupby_pcu_active', 'pcu_active'], groupby_pcuactive)
    accum(param, ['groupby_pmu_active', 'pmu_active'], groupby_pmuactive)
    # 2. Compute join
    # Determine throughput bottleneck. 
    # if R and T are small, it will be load bound on streaming S
    # if R and T are large enough, it will be compute bound
    # Total amt of compare
    # = N/(h_bkt * g_bkt) * K/h_bkt * (P * K/g_bkt + (1-P)) * h_bkt * g_bkt
    # P = (N*K/d)/(K/h_bkt * N/(g_bkt*h_bkt)*h_bkt*g_bkt) = (NK/d) / (NK/h_bkt) = h_bkt/d

    # 2.1 Load R and T on-chip

    stages = []
    accum(param, ['load_R_T_lat', 'cycle'], comp_dram_lat(param['R_size']+param['T_size'], 1, param))

    # Streaming S 
    stages.append((
        'load',
        param['S_size'], # N
        param['dram_bw_wordpcycle'], # throughput
        # fringe => pcu + compare + pcu => fringe
        param['dram_lat_cycle'] # lat
    ))

    # 2.2 Compute join
    param['comp_R_S'] = divup(param['h_bkt'] * param['g_bkt'], param['npcu']) * \
        divup(param['S_rec'], param['h_bkt'] * param['g_bkt']) * \
        divup(param['R_rec'], param['h_bkt'])
    param['hit'] = divup(param['comp_R_S'] * param['h_bkt'] , param['d'])
    param['miss'] = param['comp_R_S'] - param['hit']
    param['tot_comp_rec'] = param['hit'] * divup(divup(param['T_rec'], param['g_bkt']), param['pcu_lane']) + param['miss']
    stages.append((
        'comp',
        param['tot_comp_rec'], # N
        1, # throughput
        # fringe => pcu + compare + pcu => fringe
        param['net_lat'] + param['pcu_stage'] + param['net_lat'] # lat
    ))
    
    accum(param, ['join_onchip_lat', 'cycle'], comp_streampipe_lat('join', stages, param))
    accum(param, ['join_pcu_active', 'pcu_active'], param['tot_comp_rec']*param['npcu'])
    accum(param, ['join_pmu_active', 'pmu_active'], param['tot_comp_rec']*param['npcu'])

def derive_stat(param):
    param['sec'] = param['cycle'] * 1.0 / param['clock']
    param['min'] = param['sec'] / 60
    param['pcu_energy']=(param['pcu_active'] * 1.0 / param['clock']) * (param['pcu_power_mW'] / 1e3)
    param['pmu_energy']=(param['pmu_active'] * 1.0 / param['clock']) * (param['pmu_power_mW'] / 1e3)
    param['total_energy'] = param['pcu_energy'] + param['pmu_energy']
    param['pcu_power'] = param['pcu_energy'] / param['sec']
    param['pmu_power'] = param['pmu_energy'] / param['sec']
    param['total_power'] = param['total_energy'] / param['sec']
    param['norm_energy'] = param['total_energy'] / param['N']
    param['rec'] = param['R_rec'] + param['S_rec'] + param['T_rec']
    param['rec/s'] = param['rec'] / param['sec']

# Only set param if not already set by user
def set_param(param, **kvs):
    for k in kvs:
        if k == 'cycle':
            assert(type(kvs[k]) != float)
        if k not in param:
            param[k] = kvs[k]
        elif type(param[k]) == types.FunctionType and k != 'algo':
            param[k] = param[k](param)

def accum(param, ks, v):
    for k in ks:
        if k == 'cycle':
            assert(type(v) != float)
        if k not in param:
            param[k] = v
        else:
            param[k] += v

def init_param(**kvs):
    param = OrderedDict()
    # Overriding params
    for k in kvs:
        param[k] = kvs[k]
    
    # Arch params
    set_param(param,nrow=16)
    set_param(param,nrow=16)
    set_param(param,ncol=8)
    set_param(param,clock=1e9)
    set_param(param,pcu_stage=6)
    set_param(param,pcu_lane=16)
    set_param(param,pmu_cap_byte=256*1024)
    set_param(param,pcu_reg=64)
    set_param(param,dram_bw_GBs=49)
    set_param(param,dram_lat_cycle=43) # simulated from DRAMSim
    set_param(param,dram_capacity_TB=2)
    set_param(param,ssd_bw_GBs=0.7)
    set_param(param,pcu_area_mm2=0.849)
    set_param(param,pmu_area_mm2=0.532)
    set_param(param,pcu_power_mW=224)
    set_param(param,pmu_power_mW=300.3135)
    set_param(param,npmu=param['nrow'] * param['ncol'] / 2)
    set_param(param,npcu=param['nrow'] * param['ncol'] / 2)
    set_param(param,pmu_cap_word=divup(param['pmu_cap_byte'], 4))
    set_param(param,dram_bw_wordpcycle=param['dram_bw_GBs']*1.0*1e9/param['clock']/4)
    set_param(param,ssd_bw_wordpcycle=param['ssd_bw_GBs']*1.0*1e9/param['clock']/4)
    set_param(param,net_lat=param['nrow'] + param['ncol'])
    set_param(param,total_area=param['npmu'] * param['pmu_area_mm2'] + param['npcu'] * param['pcu_area_mm2'])

    # Algo params
    set_param(param,algo=join3)
    # Number of columns in table
    set_param(param,R_col=2) # (B,row_id)
    set_param(param,S_col=3) # (B,C,row_id)
    set_param(param,T_col=2) # (C,row_id)
    set_param(param,RS_col=4) # (B,C,R_rowid,S_rowid)
    set_param(param,ST_col=4) # (B,C,S_rowid,T_rowid)

    # set default problem size
    if param['algo'] in [join3, join32, join2, join2_slow]:
        set_param(param,d=10**9) # Number of user
        set_param(param,N=300*param['d']) # number records in R,S,T table
        assert(param['d'] <= param['N'])
        set_param(param,R_rec=param['N'])
        set_param(param,S_rec=param['N']) 
        set_param(param,T_rec=param['N'])
    elif param['algo'] in [sbsjoin2, sbsjoin3]:
        set_param(param,N=1000000) # number records in S table
        set_param(param,K=1000) # number records in R and T table
        set_param(param,d=100) # maximum number of distinct value in all table for column B 
        assert(param['d'] <= param['K'])
        assert(param['K'] <= param['N'])
        set_param(param,R_rec=param['K'])
        set_param(param,S_rec=param['N']) 
        set_param(param,T_rec=param['K'])

    set_param(param,RS_rec=divup(param['R_rec']*param['S_rec'], param['d']))

    # set default number of hash buckets
    if param['algo'] in [join3, join32]:
        set_param(param,Hop=2) # number of operations for hash function H
        set_param(param,hop=2) # number of operations for hash function h 
        set_param(param,gop=2) # number of operations for hash function g

        # h_bkt * H_bkt * g_bkt <= d. Cannot have more hash buckets more than # of unique elements
        set_param(param,h_bkt=min(param['npmu'], param['d'])) # number of bucket for h
        set_param(param,H_bkt=param['d'] / param['h_bkt']) # number of bucket for H
        set_param(param,g_bkt=param['d']) # number of bucket for g
        assert(param['h_bkt']*param['H_bkt'] <= param['d']),  \
            'h_bkt*H_bkt={},d={}'.format(param['h_bkt']*param['H_bkt'], param['d'])
        assert(param['g_bkt'] <= param['d']),  \
            'g_bkt={},d={}'.format(param['g_bkt'], param['d'])

        # Number of records after hashes
        set_hashed_rec(param, 'R', 'H')
        set_hashed_rec(param, 'RH', 'h')
        set_hashed_rec(param, 'S', 'H')
        set_hashed_rec(param, 'SH', 'h')
        set_hashed_rec(param, 'SHh', 'g')
        set_hashed_rec(param, 'T', 'g')
    elif param['algo'] in [join2, join2_slow]:
        set_param(param,gop=2)
        set_param(param,hop=2) # number of operations for hash function H
        set_param(param,Gop=2)
        set_param(param,Hop=2) # number of operations for hash function H

        set_param(param,h_bkt=min(param['npmu'], param['d'])) # number of bucket for h
        set_param(param,g_bkt=min(param['npmu'], param['d'])) # number of bucket for g
        set_param(param,H_bkt=param['d'] / param['h_bkt']) # number of bucket for H
        set_param(param,G_bkt=param['d'] / param['g_bkt']) # number of bucket for G
        assert(param['h_bkt']*param['H_bkt'] <= param['d']), \
            'h_bkt*H_bkt={},d={}'.format(param['h_bkt']*param['H_bkt'], param['d'])
        assert(param['g_bkt']*param['G_bkt'] <= param['d']), \
            'g_bkt*G_bkt={},d={}'.format(param['g_bkt']*param['G_bkt'], param['d'])

        set_hashed_rec(param, 'R', 'H')
        set_hashed_rec(param, 'RH', 'h')
        set_hashed_rec(param, 'S', 'H')
        set_hashed_rec(param, 'SH', 'h')
        set_hashed_rec(param, 'RS', 'G')
        set_hashed_rec(param, 'RSG', 'g')
        set_hashed_rec(param, 'T', 'G')
        set_hashed_rec(param, 'TG', 'g')
    elif param['algo'] in [sbsjoin2]:
        set_param(param,hop=2) # number of operations for hash function h 
        set_param(param,gop=2) # number of operations for hash function g
        set_param(param,h_bkt=min(param['npmu'], param['d'])) # number of bucket for h
        set_param(param,g_bkt=min(param['npmu'], param['d'])) # number of bucket for g
        assert(param['h_bkt'] <= param['d']),  'h_bkt={},d={}'.format(param['h_bkt'], param['d'])
        assert(param['g_bkt'] <= param['d']),  'g_bkt={},d={}'.format(param['g_bkt'], param['d'])
        assert(param['h_bkt'] <= param['npmu']), \
            'h_bkt,npmu={}'.format(param['h_bkt'], param['npmu'])
        assert(param['g_bkt'] <= param['npmu']), \
            'g_bkt,npmu={}'.format(param['g_bkt'], param['npmu'])

        set_hashed_rec(param, 'R', 'h')
        set_hashed_rec(param, 'S', 'h')
        set_hashed_rec(param, 'T', 'g')
        set_hashed_rec(param, 'RS', 'g')

    elif param['algo'] in [sbsjoin3]:
        set_param(param,hop=2) # number of operations for hash function h 
        set_param(param,gop=2) # number of operations for hash function g
        set_param(param,h_bkt=min(8, param['d'])) # number of bucket for h
        set_param(param,g_bkt=min(param['npmu']/param['h_bkt'], param['d'])) # number of bucket for g
        assert(param['h_bkt'] <= param['d']),  'h_bkt={},d={}'.format(param['h_bkt'], param['d'])
        assert(param['g_bkt'] <= param['d']),  'g_bkt={},d={}'.format(param['g_bkt'], param['d'])
        assert(param['h_bkt']*param['g_bkt'] <= param['npmu']), \
            'h_bkt*g_bkt,npmu={}'.format(param['h_bkt']*param['g_bkt'], param['npmu'])

        set_hashed_rec(param, 'R', 'h')
        set_hashed_rec(param, 'S', 'h')
        set_hashed_rec(param, 'Sh', 'g')
        set_hashed_rec(param, 'T', 'g')

    set_size(param)

    # Feasibility checking by checking bucket size in byte and compare to on-chip capacity
    if param['algo'] in [join3, join32]:
        # A size that can fit in double buffered PMU. 
        assert(param['RHh_size'] <= param['pmu_cap_word']/2), \
            "RHh_size={} > pmu_cap/2={}".format(param['RHh_size'], param['pmu_cap_word']/2)
        assert(param['SHhg_size'] <= param['pcu_reg']), \
            "SHhg_size={} > pcu_reg={}".format(param['SHhg_size'], param['pcu_reg'])
    elif param['algo'] in [join2, join2_slow]:
        # double buffered PMU will be used to store RHh
        assert(param['RHh_size'] <= param['pmu_cap_word']/2), \
            "RHh_size={} > pmu_cap/2={}".format(param['RHh_size'], param['pmu_cap_word']/2)
        # double buffered PMU will be used to store TGg
        assert(param['TGg_size'] <= param['pmu_cap_word']/2), \
            "TGg_size={} > pmu_cap/2={}".format(param['TGg_size'], param['pmu_cap_word']/2)
    elif param['algo'] in [sbsjoin2]:
        # double buffered PMU will be used to store Rh
        assert(param['Rh_size'] <= param['pmu_cap_word']/2), \
            "Rh_size={} > pmu_cap/2={}".format(param['Rh_size'], param['pmu_cap_word']/2)
        # double buffered PMU will be used to store Tg
        assert(param['Tg_size'] <= param['pmu_cap_word']/2), \
            "Tg_size={} > pmu_cap/2={}".format(param['Tg_size'], param['pmu_cap_word']/2)
    elif param['algo'] in [sbsjoin3]:
        # double buffered PMU will be used to store Rh and Tg
        assert(param['Rh_size'] <= param['pmu_cap_word']/4), \
            "Rh_size={} > pmu_cap/4={}".format(param['Rh_size'], param['pmu_cap_word']/4)
        # double buffered PMU will be used to store Tg
        assert(param['Tg_size'] <= param['pmu_cap_word']/4), \
            "Tg_size={} > pmu_cap/4={}".format(param['Tg_size'], param['pmu_cap_word']/4)

    # Performance and Power Statistics
    set_param(param,cycle=0)
    set_param(param,pcu_active=0)
    set_param(param,pmu_active=0)

    return param

def tableName(k):
    table = ''.join(filter(lambda x: x in ['R','S','T'], k))
    return table

def set_hashed_rec(param, table, h):
    param[table+h+'_rec'] = divup(param[table+'_rec'], param[h+'_bkt'])

def set_size(param):
    for k in param:
        if '_rec' in k:
            table = tableName(k)
            param[k.replace('_rec','_size')] = param[k] * param[table + '_col']
            param[k.replace('_rec','_size_TB')]=param[table+'_size']*4.0/(2**40)

# Estimate of joined table records assuming uniform distribution. 
# N1, N2 are number of records in each table. 
# d1, d2 are number of distinct value in the join column in R1 and R2
def join_rec(R1,R2,param):
    N1 = param[R1+'_rec']
    N2 = param[R2+'_rec']
    d = param['d']
    assert(d <= N1), 'd={} > N1={}'.format(d,N1)
    assert(d <= N2), 'd={} > N2={}'.format(d,N2)
    return divup(N1 * N2, d)

