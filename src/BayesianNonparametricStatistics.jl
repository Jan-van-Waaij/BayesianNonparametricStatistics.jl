__precompile__(true)

module BayesianNonparametricStatistics

export
# model.jl defines:
AbstractModel, SDEModel,
# samplepath.jl defines:
AbstractSamplePath, SamplePath, isincreasing,
# basisfunctions.jl defines:
fourier, faberschauder, faberschauderone,
# SDE.jl defines:
AbstractSDE, SDE,
# gaussianvectors.jl defines:
GaussianVector,
# gaussianprocess.jl defines:
AbstractGaussianProcess, GaussianProcess,
FaberSchauderExpansionWithGaussianCoefficients, sumoffunctions,
# defined in both gaussianvectors.jl and gaussianprocess.jl:
mean, var,
# calculateposterior.jl defines:
calculateposterior

if v"0.6" ≤ VERSION < v"0.7-"
	include(joinpath("version0dot6", "BNPversion0dot6.jl"))
elseif VERSION ≥ v"0.7-"
	include(joinpath("version0dot7and1dot0", "BNPversion0dot7and1dotX.jl"))
else
	throw(LoadError("BayesianNonparametricStatistics is only available for Julia version 0.6 and higher."))
end 
end # End of module.
