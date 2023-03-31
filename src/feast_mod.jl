
# Equations being implemented:
using JLD2
using Revise
using LinearAlgebra


# https://ieeexplore.ieee.org/document/9864144
module Feast
    using LinearAlgebra
    import LinearAlgebra.norm
    import LinearAlgebra.mul!
    using LoopVectorization
    using StaticArrays
    import InteractiveUtils.@code_warntype
    using CUDA
    abstract type FullyConnectedLayer end
    mutable struct FC{TypeArray,TypeArray2D,P,T,S,Q,R} <: FullyConnectedLayer
        #Layer input dimensions #TODO: This need not be stored. It is useless for FC

        PlaceHolderDefiner::TypeArray
        in_rows::P # Int32
        in_cols::P #Int32
        # Flattened input dimensions
        context_size::P #$Int32
        #Hyper Parameters for learning
        # Number of neurons in the layer
        nNeurons::T
        # Learning rate of the weight update
        eta::S # Float32
        # Learning rate of the threshold update
        threshEta::S
        # Threshold Open constant
        thresholdOpen::S
        # Time constant of the timesurface
        tau::S
        # TIme constant used for keeping a trace of the layer
        traceTau::S
        # Weights of the neurons

        
        w::TypeArray2D#Matrix{Float32}  ???


        #Threholds of the neurons
        thresh::TypeArray
        # Delta stores the latest input context for which the neuron fired. 
        delta::TypeArray2D # Array{Float32,2}
        # DeltaThresh stores the latest dotproduct when the neuron fired.
        deltaThresh::TypeArray # Array{Float32,1}
        # timestamps and polarity are used to calculate the timesurface. 
        # They keep track of the latest time of the event from a given input channel. 
        timestamps::TypeArray2D # Array{Float32,2}
        polarity:: TypeArray2D # Array{Float32,2}
        # WinnerTrace is allocated memory to keep track of the firing times of the neurons in the layer. 
        winnerTrace:: TypeArray # Array{Float32,1}
        # WinnerMV is also utilized to calculate the trace of the layer
        winnerMV::TypeArray # Array{Float32,1}
        # NoWinnerTrace remembers the last time at which there was no winner
        noWinnerTrace::T # Float32
        ### Pre-allocated memories for commonly calculated matrices/Vectors along the training
        # tempTrace is pre-allocated memory to populate with calculated trace of the neurons
        tempTrace::TypeArray # Array{Float32,1}
        # Event context is pre-allocated memory to hold the event contexts
        
        
        
        event_context::TypeArray2D #:: R# Array{Float32,2} ????



        # Dot Product is the pre-allocated memory to hold the dot products
        dot_prod:: TypeArray# Array{Float32,1}

        ##
        # Constructor function.
        ##
        function FC(
            PlaceHolderDefiner,
            PlaceHolderDefiner2D,
            Precision,
            input_rows,
            input_cols,
            nNeurons,
            eta,
            threshEta,
            thresholdOpen,
            tau,
            traceTau
        )

             #MArray
            #Flattened dimension size
            context_size = input_rows * input_cols 
            event_context = zeros(typeof(eta), input_rows, input_cols)#::Array{typeof(eta),2}
            #@MArray 
            #typeof(similar(m3)) == 
            #event_context = MArray{Tuple{input_rows,input_cols},typeof(eta)}#,2,9} # (final parameter is length = 9)
            w = rand(typeof(eta), context_size, nNeurons)#::typeof(PlaceHolderDefiner)#::CuArray{typeof(eta)}
            @inline for row in range(1, size(w, 2), step = 1)
                w[:, row] = w[:, row] ./ norm(w[:, row])
            end
            thresh = zeros(typeof(eta), nNeurons)#::typeof(PlaceHolderDefiner)#::Array{typeof(eta),1}
            dot_prod = zeros(typeof(eta), nNeurons)#::typeof(PlaceHolderDefiner)#::Array{typeof(eta),1}
            deltaThresh = zeros(typeof(eta), nNeurons)#::typeof(PlaceHolderDefiner)#::Array{typeof(eta),1}

            delta = zeros(typeof(eta), context_size, nNeurons)#::typeof(PlaceHolderDefiner2D)

            timestamps = (undef, (input_rows, input_cols))#::typeof(PlaceHolderDefiner) #  Array{typeof(eta),2}
            #zeros(::UndefInitializer, ::Int16, ::UInt16)
            polarity  = similar(timestamps)#::typeof(PlaceHolderDefiner2D)#(undef, input_rows, input_cols) # Array{typeof(eta),2}

            println("gets here a")
            winnerTrace = PlaceHolderDefiner(undef, nNeurons)
            println("gets here b")

            #n#oWinnerTrace = convert_precision(0.0,typeof(eta))
            #noWinnerTrace = Float32(0.0)
            winnerMV = zeros(typeof(eta), nNeurons)::PlaceHolderDefiner#::Array{typeof(eta),1}
            println("gets here c")

            tempTrace = zeros(typeof(eta), nNeurons)::PlaceHolderDefiner#::Array{typeof(eta),1}

            noWinnerTrace = convert(typeof(eta),0.0)  #convert_precision(0.0,)

            @show(typeof(noWinnerTrace))
            @show(typeof(tempTrace))
            @show(typeof(winnerMV))
            @show(typeof(winnerTrace))
            @show(typeof(polarity))
            @show(typeof(timestamps))
            @show(typeof(delta))
            @show(typeof(deltaThresh))
            @show(typeof(dot_prod))
            @show(typeof(thresh))
            @show(typeof(w))
            @show(typeof(event_context))
            @show(typeof(context_size))

            

            # ,typeof(precision),typeof(StaticArraysCore.MArray)
            new{typeof{TypeArray},typeof{TypeArray2D},typeof{Precision},typeof(input_rows),typeof(eta),typeof{TypeArray{typeof(eta)}},typeof(delta)}(
                TypeArray,
                TypeArray2D,
                Precision,
                input_rows,
                input_cols,
                context_size,
                nNeurons,
                eta,
                threshEta,
                thresholdOpen,
                tau,
                traceTau,
                w,
                thresh,
                delta,
                deltaThresh,
                timestamps,
                polarity,
                winnerTrace,
                winnerMV,
                noWinnerTrace,
                tempTrace,
                event_context,
                dot_prod,
            )

        end

    end
    """
    Reset the timer of the layer. All the time stamps and polarities will be reset. 
    All the traces will be set to 0. reset_time has the same effect as no input events for a really
    long time.
    """
    function reset_time(layer::FC)


        fill!(layer.delta, zero(eltype(layer.delta))) 
        fill!(layer.deltaThresh, zero(eltype(layer.deltaThresh)))
        fill!(layer.timestamps, typemin(eltype(layer.timestamps))) 
        fill!(layer.polarity, zero(eltype(layer.polarity))) 
        fill!(layer.winnerTrace, typemin(eltype(layer.winnerTrace)))
        fill!(layer.winnerMV, zero(eltype(layer.winnerTrace)))
        layer.noWinnerTrace = zero(eltype(layer.winnerTrace)) #typemin(Float32)
        return nothing

    end
    """
    Adds the input event to the timestamp store. Polarity is set to whatever is the decayed value at that
    channel until time ts, and then added 1 to it. 
    """
    function add_event!(layer::FC, x, y, ts)


        layer.polarity[x, y] =
            layer.polarity[x, y] * exp((layer.timestamps[x, y] - ts) / layer.tau) + 1
        layer.timestamps[x, y] = ts

        return nothing

    end
    """
    Adds a trace event/ adds the trace of a recently fired neuron. This works the same as "add_event" but 
    it is keeping track of the firing of own neurons rather than the input spikes. 
    """
    function add_trace_event(layer::FC, winner, ts)

        layer.winnerMV[winner] =
            layer.winnerMV[winner] * exp((layer.winnerTrace[winner] - ts) / layer.traceTau) + 1
        layer.winnerTrace[winner] = ts

        return nothing
    end

    function record_individual_neuron(layer::FC,neuron,t,punishFlag::Bool)
        # neuron_trace gives the decaying trace of the neuron's activity
        neuron_trace = layer.winnerMV[neuron] * exp((layer.winnerTrace[neuron] - ts) / layer.traceTau)

        if neuron_trace >= 0.1

            reward(layer, neuron)
            
        else
            # If Neuron hasn't spiked in a while and there was a recent input spike for which there was no winner
            if punishFlag
                # Reduce the threshold
                #setindex!(layer.thresh[neuron],layer.thresh[neuron] - layer.thresholdOpen)
                layer.thresh[neuron] = layer.thresh[neuron] - layer.thresholdOpen
            end
        end

        return nothing
    end
    """
    Record next layer attention signal
    """
    function record(layer::FC, ts)

        punishFlag = exp((layer.noWinnerTrace - ts) / layer.traceTau) >= 0.1

        # Record the attention signal for each neuron
        # Threads.@threads for n::Int32 = 1:layer.nNeurons
        for n in 1:layer.nNeurons
            #TODO: Recording the attention signal can be done to each neuron parallely
            record_individual_neuron(layer,n,ts,punishFlag)
        end

        return nothing
    end
    """
    Reward individual neuron 
    """
    function reward(layer::FC, neuron::Integer)
   

        layer_w_n = view(layer.w, :, neuron)
        layer_delta_n = view(layer.delta, :, neuron)

        # W = W + η*ΔW
        @turbo @. layer_w_n =  layer_w_n + layer.eta * layer_delta_n

    
        # Normalize the weights
        @turbo @. layer_w_n = layer_w_n / norm(layer_w_n)

        # Updated threshold value based on Thresh = Thresh + η*ΔThresh
        updatedThresh = layer.thresh[neuron] +
            layer.threshEta * layer.deltaThresh[neuron]
            
        # If updated threshold is less than 0, then set it to 0. (There is no point of negative thresholds)
        updatedThresh = updatedThresh < 0.0 ? zero(typeof(updatedThresh)) : updatedThresh
        layer.thresh[neuron] = updatedThresh

        return nothing
    end
    """
    Compute the normalized event context at time ts based on the timestamp and polarity stores. 
    """

    function compute_context!(layer::FC,ts)
        @turbo @. layer.event_context =
            layer.polarity *
            exp.(
                (layer.timestamps - ts) ./
                layer.tau,
            )
        layer.event_context = layer.event_context / norm(layer.event_context)
 
        return nothing
    end
    """
    Key forward function of the layer. 
        Find the normalized context
        Perform dotproduct
        Find winner neuron
        Save the delta and deltaThresh
    """
    function forward!(layer::FC, x::Integer, y::Integer, ts::AbstractFloat,winnerNeuron::Integer)
    
        # Check if it is a valid input spike
        if x < 1 || y < 1 
            # Negative winner value indicates no winner. The same convention is used everywhere
            return -1
        end
        # Add event to the timestamp and polarity stores
        add_event!(layer, x, y, ts)

        # Compute the event context
        compute_context!(layer, ts)
        
        mul!(layer.dot_prod, layer.w', view(layer.event_context, :))
        
        max_value = 0.0

        # Find the neuron with highest dot product among the neurons whose dotproduct has
        # crossed their thresholds.

        # 1D cuda kernel

        @inbounds for neuron = 1:layer.nNeurons
            if layer.dot_prod[neuron] >= layer.thresh[neuron]
                if layer.dot_prod[neuron] > max_value
                    winnerNeuron = neuron
                    max_value = getindex(layer.dot_prod,neuron)
                end
            end
        end


        if winnerNeuron > -1
            # If there is a winner, then save the delta and delta thresh which will be used later to reward 
            layer_delta = view(layer.delta, :, winnerNeuron)
            @turbo @. layer_delta = view(layer.event_context, :) -  view(layer.w, :, winnerNeuron)
            setindex!(layer.deltaThresh,winnerNeuron,layer.dot_prod[winnerNeuron] - layer.thresh[winnerNeuron])
            # Add this winning to the neuron's firing activity
            # add_trace_event(layer, winnerNeuron, ts)
            # Reward the winner neuron. 
            reward(layer, winnerNeuron)

        else
            # If there is no winner save the time when there was no winner

            @code_warntype punish!(layer)

        end        
        return winnerNeuron

    end
    """
    Key forward function of the layer. 
        Find the normalized context
        Perform dotproduct
        Find winner neuron
        Save the delta and deltaThresh
    """
    function infer(layer::FC, x::Integer, y::Integer, ts::AbstractFloat)
        
        # Check if it is a valid input spike
        if x < 1 || y < 1 
            # Negative winner value indicates no winner. The same convention is used everywhere
            return -1
        end
        # Add event to the timestamp and polarity stores
        add_event(layer, x, y, ts)

        # Compute the event context
        compute_context(layer, ts)
        
        # Find the dotproduct
        mul!(layer.dot_prod, layer.w', view(layer.event_context, :))
        winnerNeuron = -1
        max_value = 0.0

        # Find the neuron with highest dot product among the neurons whose dotproduct has
        # crossed their thresholds.


        # 1D cuda kernel

        @inbounds for neuron = 1:layer.nNeurons
            if layer.dot_prod[neuron] >= layer.thresh[neuron]
                if layer.dot_prod[neuron] > max_value
                    winnerNeuron = neuron
                    max_value = layer.dot_prod[neuron]
                end
            end
        end




        
        return winnerNeuron

    end

    function punish!(layer::FC)
        """
        Punish all the neurons in the layer
        """
        @turbo @. layer.thresh::Vector{typeof(layer.thresholdOpen)} = layer.thresh - layer.thresholdOpen

        return nothing
    end

    function saveAll(layer::FC, filename::String)
        """
        Save the weights and thresholds in a file using JLD
        """
        save(filename, "w", layer.w, "thresh", layer.thresh)
    end

    function loadAll(layer::FC, filename::String)
        """
        Load the weights and thresholds in a file using JLD
        """
        file_contents = load(filename)
        layer.w = file_contents["w"]
        layer.thresh = file_contents["thresh"]
    end
    
end # end of module
