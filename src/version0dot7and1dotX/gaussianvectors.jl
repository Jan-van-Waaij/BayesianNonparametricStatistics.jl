# GaussianVectors.jl

# Define and simulate normal vectors. 
struct GaussianVector{R<:AbstractArray{Float64, 1}, S<:AbstractArray{Float64}, 
		T<:AbstractArray{Float64}}
	mean::R
	Σ::S
	var::T
	length::Int
	numberofGaussians::Int 

	function GaussianVector(mean::R, Σ::S) where {R<:AbstractArray{Float64, 1}, S<:AbstractArray{Float64}}
		length = Base.length(mean)
		size(Σ,1) == length || throw(DimensionMismatch("Dimensions of Σ and μ don't match."))
		numberofGaussians = size(Σ,2)	
		var = Σ * Σ'
		new{R, S, typeof(var)}(mean, Σ, var, length, numberofGaussians)
	end
end

GaussianVector(Σ::T) where T<:AbstractArray{Float64} = GaussianVector(spzeros(size(Σ,1)), Σ)

rand(d::GaussianVector) = d.mean + d.Σ * randn((d.numberofGaussians, 1))

mean(d::GaussianVector) = d.mean

var(d::GaussianVector) = d.var 

length(d::GaussianVector) = d.length

scaleGV(d::T, scale::Float64) where {T<:GaussianVector} = GaussianVector(scale * d.mean, scale * d.Σ)
