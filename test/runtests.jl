module TestsForBayesianNonparametricStatistics

if v"0.6" ≤ VERSION < v"0.7-"
	include("testsforversion0dot6.jl")
elseif VERSION ≥ v"0.7-"
	include("testsforversion0dot7and1dot0.jl")
else
	error("BayesianNonparametricStatistics is only available for Julia version 0.6 and higher, no test functions available.")
end 

end # module.
