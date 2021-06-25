# calculateposterior.jl

"""
    calculatedependentfaberschauderfunctions(higestlevel::Int64)

Internal function, not exported.

Calculates which Faber-Schauder functions (ψ_i,ψ_j) have essential overlapping
support, i<j. All other combinations (ψ_i, ψ_j), i<j, have essential
nonoverlapping support and hence int_0^T ψ_i(X_t)ψ_j(X_t)dt=0, which we use in
calculategirsanovmatrix.

Returns a triple (lengthvectors, rowindices, columnindices) where
```julia
lengthvectors == length(rowindices) == length(columnindices),
```
and for i in 1:lengthvectors (ψ_rowindices[i],ψ_columnindices[i]) have
essentially overlapping support. For all i, rowindices[i] < columnindices[i].
"""
function calculatedependentfaberschauderfunctions(higestlevel::Int64)
    lengthvectors = (higestlevel+1)*2^(higestlevel+1)+1 # Correct.
    rowindices = Vector{Int}(undef, lengthvectors)
    columnindices = Vector{Int}(undef, lengthvectors)
    # Ψ_1 is dependent with all basis functions, including itself. 
    rowindices[1:2^(higestlevel+1)] .= 1
    columnindices[1:2^(higestlevel+1)] = 1:2^(higestlevel+1)
    index = 2^(higestlevel+1) + 1 # point where we are in constructing rowindices and columnindices
    println("rowindices 1-10 ", rowindices[1:10])
    println("columnindices 1-10 ", columnindices[1:10])
    println("index = ", index)
    # psi_{j,k}=psi_{2^j+k} is dependent with
    # psi_{j+d,(k-1)2^d+1},...,psi_{j+d, k2^d}, d\ge0.
    for jone in 0:higestlevel
        twotothepowerjone = 2^jone
        for kone in 1:2^jone
            ione = twotothepowerjone+kone # other index system
            for  jtwo in jone:higestlevel
                d = jtwo - jone
                twotothepowerjtwo = 2^jtwo
                for ktwo in (kone-1)*2^d+1:kone*2^d
                    rowindices[index] = ione
                    columnindices[index] = twotothepowerjtwo+ktwo
                    index += 1
                end
            end
        end
    end
    println("lengthvectors ", lengthvectors)
    println("index ", index)
    @assert lengthvectors + 1 == index 
    return (lengthvectors, rowindices, columnindices)
end

function oldcalculatedependentfaberschauderfunctions(higestlevel::Int64)
    lengthvectors = higestlevel*2^(higestlevel+1)+1 # Correct.
    rowindices = Vector{Int}(undef, lengthvectors)
    columnindices = Vector{Int}(undef, lengthvectors)
    index = 1
    # column value need to be strictly greater than row value.
    # psi_1 is dependent with every Faber-Schauder function.
    for i in 2:2^(higestlevel+1)
        rowindices[index] = 1
        columnindices[index] = i
        index += 1
    end
    println("rowindices 1-10 ", rowindices[1:10])
    println("columnindices 1-10 ", columnindices[1:10])
    println("index = ", index)
    # psi_{j,k}=psi_{2^j+k} is dependent with
    # psi_{j+d,(k-1)2^d+1},...,psi_{j+d, k2^d}, d\ge1.
    for jone in 0:higestlevel-1
        twotothepowerjone = 2^jone
        for kone in 1:2^jone
            ione = twotothepowerjone+kone
            for  jtwo in jone+1:higestlevel
                d = jtwo - jone
                twotothepowerjtwo = 2^jtwo
                for ktwo in (kone-1)*2^d+1:kone*2^d
                    rowindices[index] = ione
                    columnindices[index] = twotothepowerjtwo+ktwo
                    index += 1
                end
            end
        end
    end
    print(lengthvectors == index - 1)
    return (lengthvectors, rowindices, columnindices)
end
#test written

# calculategirsanovmatrixelement methods:

"""
    calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σ, Δt)
    calculategirsanovmatrixelement(samplevalueindices::BitArray{1},
        ψ1Xt, ψ2Xt, σ, Δt)

Internal function, not exported!

Where ψ1Xt, ψ2Xt are AbstractArray{Float64}, and σ and Δt are either an
AbstractArray{Float64} or a Float64.

Let ψ_a and ψ_b denote two basis elements.
calculategirsanovmatrixelement calculates the (a,b) element of the Girsanov
matrix defined by int_0^T ψ_a(X_t)ψ_b(X_t)/(σ^2(X_t)) dt, where ψ_a is given
by ψ1Xt (already evaluated in X) and ψ_b by ψ2Xt (already evaluated in X).
"""
function calculategirsanovmatrixelement(
        ψ1Xt::S,
        ψ2Xt::T,
        σ::Float64,
        Δt::Float64) where {S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    return Δt * sum(ψ1Xt .* ψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        ψ1Xt::R,
        ψ2Xt::S,
        σ::Float64,
        Δt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    return sum(Δt .* ψ1Xt .* ψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        ψ1Xt::R,
        ψ2Xt::S,
        σXt::T,
        Δt::Float64) where {R<:AbstractArray{Float64},
            S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    return Δt * sum(ψ1Xt .* ψ2Xt ./ (σXt.^2))
end
#test written

function calculategirsanovmatrixelement(
        ψ1Xt::Q,
        ψ2Xt::R,
        σXt::S,
        Δt::T) where {Q<:AbstractArray{Float64}, R<:AbstractArray{Float64},
            S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    return sum(Δt .* ψ1Xt .* ψ2Xt ./ (σXt.^2))
end
#test written

### 
### Rewrite
###
function calculategirsanovmatrixelement(
        samplevalueindices1::BitArray{1},
        samplevalueindices2::BitArray{1},
        ψ1Xt::S,
        ψ2Xt::T,
        σ::Float64,
        Δt::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
# what am I doing here? 
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices1::BitArray{1},
        samplevalueindices2::BitArray{1},
        ψ1Xt::R,
        ψ2Xt::S,
        σ::Float64,
        Δt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredΔt = Δt[samplevalueindices]
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices1::BitArray{1},
        samplevalueindices2::BitArray{1},
        ψ1Xt::R,
        ψ2Xt::S,
        σXt::T,
        Δt::Float64) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt ./
        (filteredσXt.^2))
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices1::BitArray{1},
        samplevalueindices2::BitArray{1},
        ψ1Xt::Q,
        ψ2Xt::R,
        σXt::S,
        Δt::T) where {Q<:AbstractArray{Float64}, R<:AbstractArray{Float64},
            S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    samplevalueindices = samplevalueindices1 .& samplevalueindices2    
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    filteredΔt = Δt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt ./ (
        filteredσXt.^2))
end
### 
### Rewrite
###


function calculategirsanovmatrixelement(
        samplevalueindices::BitArray{1},
        ψ1Xt::S,
        ψ2Xt::T,
        σ::Float64,
        Δt::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
# what am I doing here? 
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices::BitArray{1},
        ψ1Xt::R,
        ψ2Xt::S,
        σ::Float64,
        Δt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    filteredΔt = Δt[samplevalueindices]
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt) / (σ^2)
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices::BitArray{1},
        ψ1Xt::R,
        ψ2Xt::S,
        σXt::T,
        Δt::Float64) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt ./
        (filteredσXt.^2))
end
#test written

function calculategirsanovmatrixelement(
        samplevalueindices::BitArray{1},
        ψ1Xt::Q,
        ψ2Xt::R,
        σXt::S,
        Δt::T) where {Q<:AbstractArray{Float64}, R<:AbstractArray{Float64},
            S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    filteredΔt = Δt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt ./ (
        filteredσXt.^2))
end
#test written

"""
    calculateΔt(timeinterval::AbstractRange{Float64}) = step(timeinterval)
    calculateΔt(timeinterval::AbstractArray{Float64}) = timeinterval[2:end] -
        timeinterval[1:end-1]

Internal function, not exported!

Calculated Δt, returns a Float64, when timeinterval is a range object, and
otherwise the increments of the timeinterval vector.
"""
calculateΔt(timeinterval::AbstractRange{Float64}) = step(timeinterval)
calculateΔt(timeinterval::AbstractArray{Float64}) = timeinterval[2:end] -
    timeinterval[1:end-1]
# Test written.

# calculategirsanovmatrix methods:

"""
    calculategirsanovmatrix(
        Π::FaberSchauderExpansionWithGaussianCoefficients,
        samplevalueindices::Vector{BitArray{1}},
        X::SamplePath,
        ψXt::Vector{Array{Float64,1}},
        σXt::T) where {T<:Union{S, Float64} where {S<:AbstractArray{Float64}}}
    calculategirsanovmatrix(
        Π::AbstractGaussianProcess,
        X::SamplePath,
        ψXt::S,
        σXt::T) where {S<:AbstractArray{U} where {U<:AbstractArray{Float64}},
            T<:Union{Float64, V} where {V<:AbstractArray{Float64}}}

Internal function, not exported!

Calculates the Girsanov matrix. In case of a
FaberSchauderExpansionWithGaussianCoefficients-object it will make use of the
sparsity structure of the Faber-Schauder basis.
"""
function calculategirsanovmatrix(
        sizesquarematrix::Int,
        timeinterval::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{U} where U<:AbstractArray{Float64}, T<:Union{Float64, V} where V<:AbstractArray{Float64}}
    girsanovmatrix = Array{Float64}(undef, sizesquarematrix, sizesquarematrix)
    Δt = calculateΔt(timeinterval)
    for k in 1:sizesquarematrix
        for l in k:sizesquarematrix
            girsanovmatrix[k,l] = calculategirsanovmatrixelement(
                ψXt[k], ψXt[l], σXt, Δt)
        end
    end
    return Symmetric(girsanovmatrix)
end

function calculategirsanovmatrix(
        Π::FaberSchauderExpansionWithGaussianCoefficients,
        samplevalueindices::Vector{BitArray{1}},
        timeinterval::S,
        ψXt::Vector{Array{Float64,1}},
        σXt::T) where {S<:AbstractArray{Float64},
            T<:Union{U, Float64} where {U<:AbstractArray{Float64}}}
    lengthvectors, rowindices, columnindices =
        calculatedependentfaberschauderfunctions(Π.higestlevel)
    d = length(Π)
    #numberofnonzeroelements = (2*Π.higestlevel+1)*d+2 # Correct.
    V = Vector{Float64}(undef, lengthvectors)
    Δt = calculateΔt(timeinterval)
    # rowindices, columnindices = vcat(1:d, rowindices, columnindices),
    #     vcat(1:d, columnindices, rowindices)
    for i in 1:d
        V[i] = calculategirsanovmatrixelement(samplevalueindices[rowindices[i]], samplevalueindices[columnindices[i]],
            ψXt[i], ψXt[i], σXt, Δt)
    end
    for i in d+1:lengthvectors
        V[i] = calculategirsanovmatrixelement(samplevalueindices[rowindices[i]], 
            samplevalueindices[columnindices[i]], ψXt[rowindices[i]], ψXt[columnindices[i]], σXt, Δt)
    end
    # Bestaat er een symmetric version van sparse? Ja en nu toepassen! Hoe je niet meer zo te doen. Je kunt Symmetric(sparse(bla)) doen, als A upper triangular. 
    return sparse(Symmetric(sparse(rowindices, columnindices, V, d, d)))
end


function oldcalculategirsanovmatrix(
        Π::FaberSchauderExpansionWithGaussianCoefficients,
        samplevalueindices::Vector{BitArray{1}},
        timeinterval::S,
        ψXt::Vector{Array{Float64,1}},
        σXt::T) where {S<:AbstractArray{Float64},
            T<:Union{U, Float64} where {U<:AbstractArray{Float64}}}
    lengthvectors, rowindices, columnindices =
        oldcalculatedependentfaberschauderfunctions(Π.higestlevel)
    d = length(Π)
    numberofnonzeroelements = (2*Π.higestlevel+1)*d+2 # Correct.
    V = Vector{Float64}(undef, numberofnonzeroelements)
    Δt = calculateΔt(timeinterval)
    rowindices, columnindices = vcat(1:d, rowindices, columnindices),
        vcat(1:d, columnindices, rowindices)
    for i in 1:d
        V[i] = calculategirsanovmatrixelement(samplevalueindices[i],
            ψXt[i], ψXt[i], σXt, Δt)
    end
    for i in d+1:d+lengthvectors
        V[i+lengthvectors] = V[i] = calculategirsanovmatrixelement(
            samplevalueindices[rowindices[i]], samplevalueindices[columnindices[i]], ψXt[rowindices[i]],
            ψXt[columnindices[i]], σXt, Δt)
    end
    # Bestaat er een symmetric version van sparse? Ja en nu toepassen! Hoe je niet meer zo te doen. Je kunt Symmetric(sparse(bla)) doen, als A upper triangular. 
    return sparse(rowindices, columnindices, V, d, d)
end

# """
#     sumofxtimesydividedbyzsquared(x::AbstractArray{Float64}, y::AbstractArray{Float64}, z::Float64) = sum(x .* y)/(z^2)
#     sumofxtimesydividedbyzsquared(x::AbstractArray{Float64}, y::AbstractArray{Float64}, z::AbstractArray{Float64}) = sum(x .* y ./ (z.^2)

# Internal function. Not exported! 

# Calculated sum(x*y/(σ^2)) efficiently and pointwise for x and y, and if σ is a vector, also for σ. 
# """
# sumofxtimesydividedbyzsquared(x::AbstractArray{Float64}, y::AbstractArray{Float64}, z::Float64) = sum(x .* y)/(z^2)
# sumofxtimesydividedbyzsquared(x::AbstractArray{Float64}, y::AbstractArray{Float64}, z::AbstractArray{Float64}) = sum(x .* y ./ (z.^2))

"""
    calculategirsanovvectorelement(
        samplevalueindices::BitArray{1},
        ΔXt::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    calculategirsanovvectorelement(
        samplevalueindices::BitArray{1},
        ΔXt::S,
        ψXt::T,
        σ::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    calculategirsanovvectorelement(
        ΔXt::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    calculategirsanovvectorelement(
        ΔXt::S,
        ψXt::T,
        σ::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}

Internal function, used by calculategirsanovvector, not exported!

Calculates the kth Girsanov vector element int_0^T ψ_k(X_t)/(σ^2(X_t)) dX_t,
where ψ_k is the kth basis function and T is the end time.
"""
function calculategirsanovvectorelement(
    ΔXt::R,
    ψXt::S,
    σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
        T<:AbstractArray{Float64}}
    return sum(ΔXt .* ψXt ./ (σXt.^2))
end
#Test written.

function calculategirsanovvectorelement(
    ΔXt::S,
    ψXt::T,
    σ::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    return sum(ΔXt .* ψXt) / (σ^2)
end
#Test written.

function calculategirsanovvectorelement(
        samplevalueindices::BitArray{1},
        ΔXt::S,
        ψXt::T,
        σ::Float64) where {S<:AbstractArray{Float64}, T<:AbstractArray{Float64}}
    filteredΔXt = ΔXt[samplevalueindices]
    filteredψXt = ψXt[samplevalueindices]
    return sum(filteredΔXt .* filteredψXt) / (σ^2)
end
#Test written.

function calculategirsanovvectorelement(
        samplevalueindices::BitArray{1},
        ΔXt::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{Float64},
            T<:AbstractArray{Float64}}
    filteredΔXt = ΔXt[samplevalueindices]
    filterψXt = ψXt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    return sum(filteredΔXt .* filterψXt ./ (filteredσXt.^2))
end
#Test written.

"""
    calculategirsanovvector(
        lengthvector::Int,
        samplevalueindices::Vector{BitArray{1}},
        samplevalues::S,
        ψXt::Vector{Array{Float64,1}},
        σXt::T) where {S<:AbstractArray{Float64}, T<:Union{Float64, U} where
            U<:AbstractArray{Float64}}
    calculategirsanovvector(
        lengthvector::Int,
        samplevalues::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{U} where
            U<:AbstractArray{Float64}, T<:Union{Float64, V} where
            V<:AbstractArray{Float64}}

Internal function, not exported!

Calculates the Girsanov vector.
"""
function calculategirsanovvector(
        lengthvector::Int,
        samplevalues::R,
        ψXt::S,
        σXt::T) where {R<:AbstractArray{Float64}, S<:AbstractArray{U} where
            U<:AbstractArray{Float64}, T<:Union{Float64, V} where
            V<:AbstractArray{Float64}}
    ΔXt = samplevalues[2:end] - samplevalues[1:end-1]
    return [calculategirsanovvectorelement(ΔXt, ψXt[k], σXt) for k in
        1:lengthvector]
end
# Test written.

function calculategirsanovvector(
        lengthvector::Int,
        samplevalueindices::Vector{BitArray{1}},
        samplevalues::S,
        ψXt::Vector{Array{Float64,1}},
        σXt::T) where {S<:AbstractArray{Float64}, T<:Union{Float64, U} where
            U<:AbstractArray{Float64}}
    ΔXt = samplevalues[2:end] - samplevalues[1:end-1]
    return [calculategirsanovvectorelement(samplevalueindices[k], ΔXt, ψXt[k],
        σXt) for k in 1:lengthvector]
end
# Test written.

"""
    calculateσXt(σ::Float64, v::AbstractArray{Float64}) = σ
    calculateσXt(σ::Function, v::AbstractArray{Float64}) = σ.(v)

Internal function, not exported!
"""
calculateσXt(σ::Float64, v::AbstractArray{Float64}) = σ
calculateσXt(σ::Function, v::AbstractArray{Float64}) = σ.(v)
#Tests written.

"""
    calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients,
        X::SamplePath, model::SDEModel)
    calculateposterior(Π::AbstractGaussianProcess,
        X::SamplePath, model::SDEModel)::GaussianProcess

Calculates the posterior distribution Π(⋅∣X), and returns it as a
GaussianProcess object. Uses model to determine σ.

# Examples

##Example Faber-Schauder expansion
```julia
model = SDEModel(1.0, 0.0, 10000.0, 0.1)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:5])
postΠ = calculateposterior(Π, X, model)
```
##Example Fourier expansion
```julia
using Distributions
α = 0.5
model = SDEModel(1.0, 0.0, 10000.0, 0.1)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
distribution = MvNormalCanon([k^(α+0.5) for k in 1:50])
Π = GaussianProcess([fourier(k) for k in 1:50], distribution)
postΠ = calculateposterior(Π, X, model)
```
"""
function calculateposterior(Π::AbstractGaussianProcess,
        X::SamplePath, model::SDEModel)::GaussianProcess
    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]
    lengthΠ = length(Π)
    precisionprior = invcov(Π.distribution)
    girsanovvector = calculategirsanovvector(lengthΠ, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(lengthΠ, X.timeinterval, ψXt, σXt)
    precisionmatrixposterior = Matrix(girsanovmatrix) + precisionprior
    potentialposterior = girsanovvector + precisionprior * Vector(mean(Π.distribution))
    posteriordistributiononcoefficients = MvNormalCanon(potentialposterior, precisionmatrixposterior)
    return GaussianProcess(Π.basis, posteriordistributiononcoefficients)
end # Seems good. 

# Check deze functie. 
function calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients,
        X::SamplePath, model::SDEModel)
    samplevaluesmod1 = mod.(X.samplevalues[1:end-1], 1.0)
    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]
    # You only need to use the values which are in the support of the function.
    samplevalueindices = [Π.leftboundssupport[i] .≤
        samplevaluesmod1 .≤ Π.rightboundssupport[i] for i in 1:length(Π)]
    girsanovvector = calculategirsanovvector(length(Π), samplevalueindices,
        X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(Π, samplevalueindices,
        X.timeinterval, ψXt, σXt)
    oldgirsanovmatrix = oldcalculategirsanovmatrix(Π, samplevalueindices,
        X.timeinterval, ψXt, σXt)
    return oldgirsanovmatrix, girsanovmatrix
    println("Same matrix? ", girsanovmatrix == oldgirsanovmatrix) ## De oude en de nieuwe matrix zijn niet hetzelfde. 
    println(girsanovmatrix)
    println(oldgirsanovmatrix)
    precisionprior = invcov(Π.distribution)
    precisionmatrixposterior = girsanovmatrix + precisionprior
    println("Symmetric? ", issymmetric(precisionmatrixposterior))
    println("Positive definite? ", isposdef(precisionmatrixposterior)) # Hier is een issue: the precisionmatrixposterior is not positive definite. 
    potentialposterior = girsanovvector + precisionprior * Vector(mean(Π.distribution))
    posteriordistributiononcoefficients = MvNormalCanon(potentialposterior, precisionmatrixposterior)
    return FaberSchauderExpansionWithGaussianCoefficients(Π.higestlevel, posteriordistributiononcoefficients)
end

function calculateposteriorcoefficients(d::GaussianVector,
        X::SamplePath, model::SDEModel)::GaussianProcess
    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]
    lengthΠ = length(d)
    covariancematrixprior = Matrix(cov(Π.distribution))
    precisionprior = inv(covariancematrixprior)
    girsanovvector = calculategirsanovvector(lengthΠ, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(lengthΠ, X.timeinterval, ψXt, σXt)
    precisionmatrixposterior = Matrix(girsanovmatrix) + precisionprior
    potentialposterior = girsanovvector + precisionprior * Vector(mean(Π.distribution))
    meanposterior =  covariancematrixprior * potentialposterior
    posteriordistribution = GaussianVector(meanposterior, 
        precisionmatrixposterior^(-0.5))
    return GaussianProcess(Π.basis, posteriordistribution)
end

function GibbsIter(siteration, A, B, meanGPPost, precGPPost, numberofbasisfunctions, X, ψXt, σXt)
    meanGPPost /= sqrt(siteration)
    precGPPost *= sqrt(siteration)
    girsanovvector = calculategirsanovvector(numberofbasisfunctions, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(numberofbasisfunctions, X.timeinterval, ψXt, σXt)
    precGPPost += Matrix(girsanovmatrix)
    # Lijn hieronder moet de rekenfout hersteld worden.
    potentialGPPost = girsanovmatrix + meanGPPost
    meanGPPost = precGPPost \ potentialGPPost
    distGPPost = GaussianVector(meanGPPost, sqrt(inv(precGPPost)))
    ϴ = rand(distGPPost)
    siteration = rand(Gamma(A+numberofbasisfunctions/2, B + 0.5 * (ϴ - meanGPPost)' * 
        potentialGPPost * (ϴ - meanGPPost)))
    return (siteration, meanGPPost, precGPPost, ϴ)
end 


function GibbsSampler(Π::S, A::Float64, B::Float64, X::SamplePath, model::SDEModel,
        numSamples::Int, S0 = 1.0, burnin::Int = 100, intercept::Int = 1) where {S<:AbstractGaussianProcess}
    meanGPPost = mean(Π.distribution)
    precGPPost = inv(Matrix(cov(Π.distribution)))
    numberofbasisfunctions = convert(Float64, length(Π))

    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]

    siteration = S0

    for i in 1:burnin
        siteration, meanGPPost, precGPPost = GibbsIter(siteration, A, B, meanGPPost, 
            precGPPost, numberofbasisfunctions, X, ψXt, σXt)
    end

    samplePostS = Array{Float64}(undef, numSamples)
    samplePostϴ = Array{Float64}(undef, numSamples, numberofbasisfunctions)
    for i in 1:numSamples
        for j in 1:(intercept+1)
            siteration, meanGPPost, precGPPost, ϴ = GibbsIter(siteration, A, B, 
                meanGPPost, precGPPost, numberofbasisfunctions, X, ψXt, σXt)
        end
        samplePostS[i], samplePostϴ[i,:] = siteration, ϴ
    end
    return (samplePostS, samplePostϴ)
end 





