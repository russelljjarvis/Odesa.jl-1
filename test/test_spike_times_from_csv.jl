using CSV
using Odesa
using CSV, DataFrames
using Revise
using Plots
import Plots.plot
using StatsBase
using Test
using CUDA

function get_isis(times,nodes)
    spike_dict = Dict()
    all_isis = []
    for n in unique(nodes)
        spike_dict[n] = []
    end
    for (st,n) in zip(times,nodes)
        append!(spike_dict[n],st)
    end
    for (k,v) in pairs(spike_dict)
        time_old = 0
        for time in spike_dict[k][1:end-1]
            isi = time - time_old
            append!(all_isis,isi)
            time_old = time
        end
    end
    return StatsBase.mean(all_isis)
end

function get_layer(inv_isi::AbstractFloat, precisionF::Type, precisionInt::Type;cuda=false) 
    layer_nNeurons::precisionInt = 20
    layer_eta::precisionF = 0.001
    layer_threshEta::precisionF = 0.001
    layer_thresholdOpen::precisionF = 0.01
    layer_tau::precisionF =  convert(precisionF,inv_isi) 
    layer_traceTau::precisionF = 0.81
    precision::precisionF = convert(precisionInt,0)  

    if cuda
        TypeArray = CuArray{typeof(precision),1}(zeros(typeof(layer_eta), layer_nNeurons))
        TypeArray2D = CuArray{typeof(precision),2}(zeros(typeof(layer_eta), (layer_nNeurons,2)))
        layer = Odesa.Feast.FC(TypeArray,TypeArray2D,precision,precisionInt(1),precisionInt(pop_size),layer_nNeurons,layer_eta,layer_threshEta,layer_thresholdOpen,layer_tau,layer_traceTau)
    else
        TypeArray = Array{typeof(precision),1}(zeros(typeof(layer_eta), layer_nNeurons))
        TypeArray2D = Array{typeof(precision),2}(zeros(typeof(layer_eta), (layer_nNeurons,2)))
        layer = Odesa.Feast.FC(TypeArray,TypeArray2D,precision,precisionInt(1),precisionInt(pop_size),layer_nNeurons,layer_eta,layer_threshEta,layer_thresholdOpen,layer_tau,layer_traceTau)
    end
    return layer
end
df = CSV.read("times_for_yesh.csv",DataFrame)

nodes = df.x1
times = df.x2
perm = sortperm(times)
nodes = convert(Vector{UInt64},nodes[perm])
times = times[perm]
inv_isi = Float32(1.0/get_isis(times,nodes))
layer16 = get_layer(inv_isi,Float16,UInt16,cuda=true)
layer32 = get_layer(inv_isi,Float32,UInt32)
layer16 = get_layer(inv_isi,Float16,UInt16)

winners = []
#p1=plot(layer.thresh)
function collect_distances(layer,nodes,times,precisionF,precisionInt)
    distances = layer.dot_prod
    winnerNeuron = -1
    @inbounds for i in 1:325
        Odesa.Feast.reset_time(layer)
        @inbounds for (y,ts) in zip(nodes,times)
            Odesa.Feast.forward!(layer, precisionInt(1), precisionInt(y), precisionF(ts), winnerNeuron)    
            @show(winnerNeuron)
            distances = layer.dot_prod
            
        end
        #display(plot!(p1,layer.thresh,legend=false))
    end
    distances
end
@time distances = collect_distances(layer16,nodes,times,Float16,UInt16)
@time distances = collect_distances(layer32,nodes,times,Float32,UInt32)
@time distances = collect_distances(layer16,nodes,times,Float16,UInt16)


@assert length(distances) !=0
@assert mean(distances) !=0


"""
Get time surfaces for plotting.
"""
function get_ts(nodes,times)
    # The temporal resolution of the final timesurface
    dt = 10
    num_neurons = Int(length(unique(nodes)))+1#int(df.max(axis=0)['x1'])
    total_time =  Int(maximum(times))
    time_resolution = Int(round(total_time/dt))
    # Final output. 
    final_timesurf = zeros((num_neurons, time_resolution+1))
    # Timestamp and membrane voltage store for generating time surface
    timestamps = zeros((num_neurons)) .- Inf
    mv = zeros((num_neurons))
    tau = 200
    last_t = 0
    for (tt,nn) in zip(times,nodes)
        #Get the current spike
        neuron = Int(nn) 
        time = Int(tt)        
        # If time of the next spikes leaps over, make sure to generate 
        # timesurfaces for all the intermediate dt time intervals and fill the 
        # final_timesurface.
        if time > last_t
            timesurf = similar(final_timesurf[:,1])
            for t in collect(last_t:dt:time)
                @. timesurf = mv*exp((timestamps-t)/tau)
                final_timesurf[:,1+Int(round(t/dt))] = timesurf
            end
            last_t = time
        end
        # Update the membrane voltage of the time surface based on the last value and time elapsed
        mv[neuron] =mv[neuron]*exp((timestamps[neuron]-time)/tau) +1
        timestamps[neuron] = time
        # Update the latest timestamp at the channel. 
    end
    # Generate the time surface for the rest of the time if there exists no other spikes. 
    timesurf = similar(final_timesurf[:,1])
    for t in collect(last_t:dt:total_time)
        @. timesurf = mv*exp((timestamps-t)/tau)
        final_timesurf[:,1+Int(round(t/dt))] = timesurf
    end
    return final_timesurf

end
final_timesurf = get_ts(nodes,times);
@show(sum(final_timesurf))
display(Plots.heatmap(final_timesurf))
@assert mean(final_timesurf) !=0
