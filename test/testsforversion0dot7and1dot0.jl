using Test, Random, InteractiveUtils, SparseArrays, LinearAlgebra, StatsBase

using Main.BayesianNonparametricStatistics

Base.Libc.srand(123)
e = Base.MathConstants.e
sqrttwo = sqrt(2.0)

@testset "BayesianNonparametricStatistics package" begin
    @testset "model.jl" begin
        @test supertype(SDEModel) == AbstractModel
        @test supertype(AbstractModel) == Any
        for σ in -2.0:0.1:2.0
            for beginvalue in -2.0:0.1:2.0
                for endtime in 0.1:0.1:2.0
                    for Δ in 0.01:0.01:0.99
                        M = SDEModel(σ, beginvalue, endtime, Δ)
                        @test M.σ == σ
                        @test M.beginvalue == beginvalue
                        @test M.endtime == endtime
                        @test M.Δ==Δ
                    end
                end
            end
        end
        σ = identity
        M = SDEModel(σ, 0.0,10.0, 0.01)
        @test M.σ == identity
        σ(x) = 1.0
        M = SDEModel(σ, 0.0,10.0,0.01)
        @test M.σ == σ

        σ(x) = 2+sin(x)
        M = SDEModel(σ, 1.3, 101.0, 0.5)
        @test M.σ == σ
        @test M.beginvalue == 1.3
        @test M.endtime == 101.0
        @test M.Δ == 0.5

        @test_throws ArgumentError SDEModel(sin, 0.0, -1.0, 0.1)
        @test_throws ArgumentError SDEModel(sin, 0.0, 1.0, -0.1)
        @test_throws ArgumentError SDEModel(sin, 0.0, -1.0, -0.1)
        @test_throws ArgumentError SDEModel(1.0, 0.0, 0.0, 0.1)
        @test_throws ArgumentError SDEModel(1.0, 0.0, 10.0, 0.0)
    end

    #Test functions are correct, complete and the same in both versions. 
    @testset "samplepath.jl" begin
        @test supertype(SamplePath) == AbstractSamplePath

        @test supertype(AbstractSamplePath) == Any

        X = SamplePath(0.0:0.1:1.0, sin)
        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == sin.(0.0:0.1:1.0)

        X = SamplePath(0.0:0.1:1.0, 1.0:11.0)

        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == 1.0:11.0

        X = SamplePath(0.0:0.1:1.0, x->2*x)
        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == collect(0.0:0.2:2.0)

        @test step(X) == 0.1
        @test length(X) == 11

        X = SamplePath([1., 2.], [1.1, 2.1])
        @test X.timeinterval == [1., 2.]
        @test X.samplevalues == [1.1, 2.1]

        @test length(X) == 2

        X = SamplePath([1.0, 3.0, 5.0, 9.0], x->x^2)
        @test X.timeinterval == [1.0, 3.0, 5.0, 9.0]
        @test X.samplevalues == [1.0, 9.0, 25.0, 81.0]

        X = SamplePath(collect(-2.0:0.01:2.0), sin)
        X.timeinterval == collect(-2.0:0.01:2.0)
        X.samplevalues == sin.(collect(-2.0:0.01:2.0))

        X = SamplePath(-2.0:0.01:2.0, sin)
        X.timeinterval == collect(-2.0:0.01:2.0)
        X.samplevalues == sin.(collect(-2.0:0.01:2.0))

        X = SamplePath(0.0:0.1:2π, sin)
        @test length(X) == length(0.0:0.1:2π)

        @test_throws DimensionMismatch SamplePath(0.0:2.0, [1.0, 2.0])

        @test_throws ArgumentError SamplePath(0.0:-0.1:-0.1, [0.0, 2.0])

        @test_throws DimensionMismatch SamplePath([0.1, 2.0],[3.0])

        @test_throws ArgumentError SamplePath([0.1, 2.0, 1.0], [1.0,2.0,3.0])

        @test BayesianNonparametricStatistics.isincreasing(0:0.1:1)
        @test !BayesianNonparametricStatistics.isincreasing(0:-0.1:-1.0)
        @test !BayesianNonparametricStatistics.isincreasing([0.0, 0.1, 0.2, 0.1, 0.3, 0.4])
        @test !BayesianNonparametricStatistics.isincreasing([0.0, -0.1, 0.0, 0.2, 0.3])
        @test !BayesianNonparametricStatistics.isincreasing([1, 2, 3, 4, -10, 11, 12])
        @test BayesianNonparametricStatistics.isincreasing([1, 2, 3, 4, 500, 600, 2000, 10000])
        @test BayesianNonparametricStatistics.isincreasing([1.0, 2.0, 2.00000001, 2.00000002])
        @test !BayesianNonparametricStatistics.isincreasing([2.0,2.0,2.0,2.0,2.0])
        @test !BayesianNonparametricStatistics.isincreasing([2.0,2.0,2.1,2.2,2.3])
        @test !BayesianNonparametricStatistics.isincreasing([2.0,2.1,2.2,2.2,2.3])
        @test !BayesianNonparametricStatistics.isincreasing([2.1,2.2,2.3,2.4,2.4])
        @test !BayesianNonparametricStatistics.isincreasing([2.0,2.0, 3.0, 3.0001])
    end

    #Test functions are correct, complete and the same in both versions. 
    @testset "basisfunctions.jl" begin
        @test_throws AssertionError fourier(0)
        @test_throws AssertionError fourier(-1)
        @test_throws AssertionError fourier(-10)

        numberofbasisfunctionstested = 100
        numberoffaberschauderlevelstested = convert(Int64, round(log2(numberofbasisfunctionstested), RoundUp)) - 1
        numberoffourierfunctionstested = numberofbasisfunctionstested
        #The Fourier functions are orthogonal.
        t = 0.0:1/(4*ceil(numberoffourierfunctionstested/2)):1.0
        tprecise = 0.0:10.0^-1/(4*ceil(numberoffourierfunctionstested/2)):1.0
        for k in 1:numberoffourierfunctionstested-1
            for l in k+1:numberoffourierfunctionstested
                phikvalues = fourier(k).(tprecise[1:end-1])
                philvalues = fourier(l).(tprecise[1:end-1])
                @test abs(sum(step(tprecise)*phikvalues.*philvalues)) < 10.0^-10
            end
        end
        # And the square integrates to one.
        for k in 1:numberoffourierfunctionstested
            values = fourier(k).(tprecise)
            @test abs(step(tprecise)*sum(values.*values)-1.0) < 10.0^-2
        end

        # They are one-periodic.
        for k in 1:numberoffourierfunctionstested
            for y in t
                value = fourier(k)(y)
                for m in -10.0:10.0
                    valuetwo = fourier(k)(y+m)
                    @test abs(value - valuetwo) < 10.0^-10
                end
            end
        end
        #
        # # They have minimum -sqrt(2) and maximum sqrt(2).
        for k in 1:numberoffourierfunctionstested
            if k % 2 == 1
                t = 0.0:0.1*1/(2k+2):1.0
            else
                t = 0.0:0.1*1/k:1.0
            end
            values = fourier(k).(t)
            @test abs(minimum(values)+sqrttwo) < 10.0^-5
            @test abs(maximum(values)-sqrttwo) < 10.0^-5
        end
        #
        # # Functions \phi_k with k odd are zero when x=m/(k+1), m=0,...,k.
        # # Functions \phi_k with k even are zero when x=(1+2m)/(2k), m=0,...,k-1.
        for k in 1:2:numberoffourierfunctionstested
            for m in 0:k
                @test abs(fourier(k)(m/(k+1))) < 10.0^-5
            end
        end
        #
        for k in 2:2:numberoffourierfunctionstested
            for m in 0:k-1
                @test abs(fourier(k)((1+2m)/(2k))) < 10.0^-5
            end
        end
        #
        # # When k is odd, then \phi_k(x)=\sqrt(2) when x = (4m+1)/(2k+2), m=0,..,(k-1)/2.
        for k in 1:2:numberoffourierfunctionstested
            for m in 0:convert(Int64,(k-1)/2)
                @test abs(fourier(k)((4m+1)/(2k+2))-sqrt(2)) < 10.0^-5
            end
        end
        #
        # When k is odd, then \phi_k(x)= -sqrt(2) when x = (3+4m)/(2k+2), m=0,..,(k-1)/2.
        for k in 1:2:numberoffourierfunctionstested
            for m in 0:convert(Int64, (k-1)/2)
                @test abs(fourier(k)((3+4m)/(2k+2))+sqrt(2)) < 10.0^-5
            end
        end
        #
        # When k is even, then \phi_k(x)=sqrt(2) when x = 2m/k, m=0,...,k/2-1.
        for k in 2:2:numberoffourierfunctionstested
            for m in 0:convert(Int64, k/2)-1
                @test abs(fourier(k)(2m/k)-sqrt(2)) < 10.0^-5
            end
        end
        #
        # when k is even, then \phi_k(x)=-sqrt(2) when x=(2m+1)/k, m = 0,...,k/2-1.
        for k in 2:2:numberoffourierfunctionstested
            for m in 0:convert(Int64, k/2)-1
                @test abs(fourier(k)((2m+1)/k)+sqrt(2)) < 10.0^-5
            end
        end
        #
        @test_throws AssertionError faberschauder(-1, 1)
        @test_throws AssertionError faberschauder(-3, 1)
        @test_throws AssertionError faberschauder(1, 0)
        @test_throws AssertionError faberschauder(1, -2)
        @test_throws AssertionError faberschauder(1, 3)

        # # Below we test Faber-Schauder functions on their mathematical properities, up
        # # to level numberoffaberschauderlevelstested, defined below.
        t = 0.:2.0^(-numberoffaberschauderlevelstested-2):1.0

        # Faber-Schauder functions are level-wise orthogonal, for levels j≥1.
        for j in 1:numberoffaberschauderlevelstested
            for k in 1:2^j-1
                for l in k+1:2^j
                    psijkvalues = faberschauder(j,k).(t[1:end-1])
                    psijlvalues = faberschauder(j,l).(t[1:end-1])
                    @test abs(step(t)*sum(psijkvalues.*psijlvalues))<10.0^-3
                end
            end
        end
        #
        # They integrate to 2^(-j-1)
        psionevalues = faberschauderone.(t[1:end-1])
        @test abs(step(t)*sum(psionevalues)-2.0^-1) < 10.0^-3
        for j in 1:numberoffaberschauderlevelstested
            for k in 1:2^j
                psijkvalues = faberschauder(j,k).(t[1:end-1])
                @test abs(step(t)*sum(psijkvalues)-2.0^(-j-1)) < 10.0^-3
            end
        end

        # psi_1 has maximum 1 at 0 and 1.
        @test abs(faberschauderone(0.0)-1.0)<10.0^-5
        @test abs(faberschauderone(1.0)-1.0)<10.0^-5

        # psi_1 has minimum 0 at 0.5.
        @test abs(faberschauderone(0.5)) < 10.0^-5

        # psi_{j,k} has maximum 1.0 at 2.0^(-j-1)+(k-1)*2.0^-j
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                @test abs(faberschauder(j,k)(2.0^(-j-1)+(k-1)*2.0^-j)-1.0)<10.0^-5
            end
        end

        #psi_{j,k} has minimum 0.0 at (k-1)*2.0^-j and k*2.0^-j
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                @test abs(faberschauder(j,k)((k-1)*2.0^-j))<10.0^-5
                @test abs(faberschauder(j,k)(k*2.0^-j))<10.0^-5
            end
        end

        # They are one-periodic.
        x = 0.0:0.01:0.99
        for y in x
            value = faberschauderone(y)
            for k in -10.0:10.0
                valuetwo = faberschauderone(y+k)
                @test abs(value-valuetwo) < 10.0^-5
            end
        end

        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                for y in x
                    value = faberschauder(j,k)(y)
                    for m in -10.:10.
                        valuetwo = faberschauder(j,k)(y+m)
                        @test abs(value-valuetwo) < 10.0^-5
                    end
                end
            end
        end
        # End of tests for mathematical properities of Faber-Schauder functions.
    end

    #Test functions are correct, complete and the same in both versions. 
    @testset "SDE.jl" begin
        @test supertype(SDE) == AbstractSDE

        @test supertype(AbstractSDE) == Any

        # The following should represent a Brownian motion.
        model = SDEModel(1.0, 0.0, 1.0, 0.001)
        sde = SDE(x->0.0, model)

        lengthvector = 1000
        x = Vector{Float64}(undef, lengthvector)
        
        for k in 1:lengthvector
            X = rand(sde)
            x[k] = X.samplevalues[end]
        end

        @test abs(StatsBase.mean(x)) < 0.1
        @test 0.9 < StatsBase.var(x) < 1.1

        X = rand(sde)

        @test length(X) == length(0.0:sde.model.Δ:sde.model.endtime)
        @test X.timeinterval == 0.0:sde.model.Δ:sde.model.endtime

        model = SDEModel(identity, 1.0, 1.0, 0.001)
        geometricbrownianmotion = SDE(x->0.5*x, model)

        for k in 1:lengthvector
            X = rand(geometricbrownianmotion)
            x[k] = X.samplevalues[end]
        end

        @test abs(e^(0.5)-StatsBase.mean(x)) < 0.1

        X = rand(geometricbrownianmotion)

        @test length(X) == length(0.0:geometricbrownianmotion.model.Δ:geometricbrownianmotion.model.endtime)
        @test X.timeinterval ==  0.0:geometricbrownianmotion.model.Δ:geometricbrownianmotion.model.endtime

        sde = SDE(x->0.0, SDEModel(1.0, 0.0, 1.0, 0.001))

        for k in 1:lengthvector
            X = rand(sde)
            x[k] = X.samplevalues[end]
        end

        @test abs(StatsBase.mean(x)) < 0.1
        @test 0.9 < StatsBase.var(x) < 1.1

        X = rand(sde)
        @test X.timeinterval == 0.0:sde.model.Δ:sde.model.endtime
        @test length(X) == length(0.0:sde.model.Δ:sde.model.endtime)

        ornsteinuhlenbeckprocess = SDE(x-> -x, SDEModel(1.0, 2.72, 1.0, 0.001))

        for k in 1:lengthvector
            X = rand(ornsteinuhlenbeckprocess)
            x[k] = X.samplevalues[end]
        end

        @test abs(StatsBase.mean(x) - 1) < 0.1

        X = rand(ornsteinuhlenbeckprocess)

        @test X.timeinterval == 0.0:ornsteinuhlenbeckprocess.model.Δ:ornsteinuhlenbeckprocess.model.endtime
        @test length(X) == length(0.0:ornsteinuhlenbeckprocess.model.Δ:ornsteinuhlenbeckprocess.model.endtime)

        model = SDEModel(1.0, 0.0, 10.0, 1.0)
        prevXval = 1.0
        @test BayesianNonparametricStatistics.calculatenextsamplevalue(prevXval, identity, model, 1.0) == 3.0

        model = SDEModel(identity, 0.0, 10.0, 1.0)
        prevXval = 1.0
        @test BayesianNonparametricStatistics.calculatenextsamplevalue(prevXval, identity, model, 1.0) == 3.0
        # End tests "sde.jl".
    end
    
    #Test functions are correct, complete and the same in both versions.
    @testset "gaussianvectors.jl" begin
        m = [10.0, -3.5]
        A = [1.0 3.0; 5.0 7.0]
        B = [1.0 1.0 1.0; 2.0 2.0 2.0 ; 3.0 4.0 5.0]
        C = m
        GV = GaussianVector(A)
        @test mean(GV) == GV.mean == [0.0, 0.0]
        @test var(GV) == GV.var == A * A'
        @test GaussianVector(C).mean ≈ [0.0, 0.0]
        GV = GaussianVector(m, A)
        @test mean(GV) ==  GV.mean == m 
        @test var(GV) == GV.var == A * A'
        @test length(GV) == GV.length == 2
        @test GV.numberofGaussians == 2
        @test_throws DimensionMismatch GaussianVector([0.0], A)
        @test_throws DimensionMismatch GaussianVector([3.4, 2.2, 9.5], A)
        @test_throws DimensionMismatch GaussianVector([0.0], [1.0, 2.0])
        @test_throws DimensionMismatch GaussianVector([0.0, 0.0], [2.0])
        GV = GaussianVector([0.0, 0.0], [1.0, 2.0])
        @test mean(GV) == GV.mean == [0.0, 0.0]
        @test var(GV) == GV.var == [1.0, 2.0] * [1.0, 2.0]'
        @test length(GV) == GV.length == 2
        @test GV.numberofGaussians == 1
        randomvector = rand(GV)
        @test 2 * randomvector[1] ≈ randomvector[2]
        m = [1.0, 2.0]
        Σ = [1.0 0.0; 0.0 2.0]
        n = 100000
        GV = GaussianVector(m, Σ)
        nrandomdraws = Array{Float64, 2}(undef, n, 2)
        for k in 1:n
            nrandomdraws[k, :] = rand(GV)
        end
        mempirical = StatsBase.mean(nrandomdraws, dims = 1)
        vempirical = StatsBase.var(nrandomdraws, dims = 1) 
        @test norm([1.0 2.0]-mempirical) < 0.1
        @test norm(vempirical-[1.0 4.0]) < 0.1
    end


    #Test functions are correct, complete and the same in both versions. 
    @testset "gaussianprocess.jl" begin
        @test Set([GaussianProcess, FaberSchauderExpansionWithGaussianCoefficients]) ⊆ Set(subtypes(AbstractGaussianProcess))

        @test supertype(AbstractGaussianProcess) == Any

        @test_throws DimensionMismatch GaussianProcess([fourier(k) for k in 1:2], GaussianVector(sparse(Diagonal([1.0,1.0,1.0]))))

        X = GaussianProcess([sinpi, cospi], GaussianVector(sparse(Diagonal([1.0, 1.0]))))

        @test X.basis == [sinpi, cospi]

        @test typeof(X.distribution) <: GaussianVector
        @test X.distribution.Σ == [1.0 0.0; 0.0 1.0]
        @test mean(X.distribution) == [0.0,0.0]
        @test length(X) == 2
        @test typeof(rand(X)) <: Function
        
        n = 100000
        a = sinpi(1/4)
        x = Vector{Float64}(undef, n) 
        for k in 1:n
            f = rand(X)
            x[k] = f(1/4)
        end

        @test abs(StatsBase.mean(x))<0.01
        @test abs(StatsBase.var(x)-2*a^2)<0.1

        x = 0.0:0.1:1.0
        y = sumoffunctions([sin, cos], [1., 1.]).(x)
        z = map(x-> sin(x)+cos(x), x)
        @test y ≈ z

        A = sparse(Diagonal([1.0, 1.0]))

        distribution = GaussianVector(A)

        @test_throws AssertionError FaberSchauderExpansionWithGaussianCoefficients(1,distribution)

        @test_throws AssertionError FaberSchauderExpansionWithGaussianCoefficients(Vector{Float64}(undef, 0))

        Π = FaberSchauderExpansionWithGaussianCoefficients(0,distribution)
        @test length(Π) == 2

        @test_throws AssertionError sumoffunctions([sin], [1.0,2.0])

        vectoroffunctions = [sin, x -> cos(x+π/2)]
        vectorofscalars = [1.0, 1.0]
        f = sumoffunctions(vectoroffunctions, vectorofscalars)
        x = 0.0:0.01:1.0
        y = f.(x)
        @test maximum(abs.(y)) < 0.001

        n = 10000
        x = Vector{Float64}(undef, n)

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.5)
        end

        @test abs(StatsBase.mean(x))<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.0)
        end

        @test abs(StatsBase.mean(x))<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(1.0)
        end

        @test abs(StatsBase.mean(x))<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        distribution = GaussianVector([1.0, -2.0], A)
        Π = FaberSchauderExpansionWithGaussianCoefficients(0,distribution)

        @test length(Π) == 2

        n = 10000
        x = Vector{Float64}(undef, n)

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.5)
        end

        @test abs(StatsBase.mean(x)+2.0)<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.0)
        end

        @test abs(StatsBase.mean(x)-1.0)<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(1.0)
        end

        @test abs(StatsBase.mean(x)-1.0)<0.1
        @test abs(StatsBase.var(x)-1.0) <0.1

        Π = FaberSchauderExpansionWithGaussianCoefficients([1.0, sqrt(2)])

        n = 10000
        x = Vector{Float64}(undef, n)
        for k in 1:n
            f = rand(Π)
            x[k] = f(0.25)
        end

        @test abs(StatsBase.mean(x)) < 0.1
        @test abs(StatsBase.var(x) - 2.5) < 0.1

        A = SparseArrays.sparse(1.0*I, 4, 4)

        distribution = GaussianVector([1.0,1.0,1.0,1.0], A)

        Π = FaberSchauderExpansionWithGaussianCoefficients(1, distribution)

        n = 10000
        x = Vector{Float64}(undef, n)
        for k in 1:n
            f = rand(Π)
            x[k] = f(0.25)
        end

        @test abs(StatsBase.mean(x)-2.0) < 0.1
        @test abs(StatsBase.var(x)-1.5) < 0.1
    end #testset "gaussianprocess.jl"

    #Test functions are correct, complete and the same in both versions. 
    @testset "calculateposterior.jl" begin
        # Test for two cases whether calculatedependentfsfunctions
        # calculates the right dependency structure of the Faber-Schauder basis.
        A = [
        true true true true true true true true;
        true true true true true true true true;
        true true true false true true false false;
        true true false true false false true true;
        true true true false true false false false;
        true true true false false true false false;
        true true false true false false true false;
        true true false true false false false true
        ]

        A = Symmetric(A)
        f = BayesianNonparametricStatistics.calculatedependentfaberschauderfunctions
        lengthvectors, rowindices, columnindices = f(2)

        @test lengthvectors == length(rowindices) == length(columnindices)

        B = falses(8, 8)
        for i in 1:lengthvectors
            B[rowindices[i], columnindices[i]] = true
            B[columnindices[i], rowindices[i]] = true
        end
        for i in 1:8
            B[i,i] = true
        end

        @test A == B

        A = [
        true true true true true true true true true true true true true true true true; #1
        true true true true true true true true true true true true true true true true; #2
        true true true false true true false false true true true true false false false false; #3
        true true false true false false true true false false false false true true true true; #4
        true true true false true false false false true true false false false false false false; #5
        true true true false false true false false false false true true false false false false; #6
        true true false true false false true false false false false false true true false false; #7
        true true false true false false false true false false false false false false true true; #8
        true true true false true false false false true false false false false false false false; #9
        true true true false true false false false false true false false false false false false; #10
        true true true false false true false false false false true false false false false false; #11
        true true true false false true false false false false false true false false false false; #12
        true true false true false false true false false false false false true false false false; #13
        true true false true false false true false false false false false false true false false; #14
        true true false true false false false true false false false false false false true false; #15
        true true false true false false false true false false false false false false false true #16
        ]

        A = Symmetric(A)

        B = falses(16, 16)
        lengthvectors, rowindices, columnindices = f(3)

        @test lengthvectors == length(rowindices) == length(columnindices)

        for i in 1:lengthvectors
            B[rowindices[i], columnindices[i]] = true
            B[columnindices[i], rowindices[i]] = true
        end
        for i in 1:16
            B[i,i] = true
        end

        @test A == B

        f = BayesianNonparametricStatistics.calculategirsanovmatrixelement

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:3.0),
            collect(1.0:2.0), collect(1.0:2.0))

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:3.0),
            collect(1.0:2.0), 3.0)

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:3.0),
            2.5, collect(1.0:2.0))

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:3.0),
            0.9, 1.2)

        @test_throws DimensionMismatch f(collect(1.0:3.0), collect(1.0:2.0),
            collect(1.0:2.0), collect(1.0:2.0))

        @test_throws DimensionMismatch f(collect(1.0:3.0), collect(1.0:2.0),
            collect(1.0:2.0), 3.0)

        @test_throws DimensionMismatch f(collect(1.0:3.0), collect(1.0:2.0),
            2.5, collect(1.0:2.0))

        @test_throws DimensionMismatch f(collect(1.0:3.0), collect(1.0:2.0),
            0.9, 1.2)

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:2.0),
            collect(1.0:3.0), collect(1.0:2.0))

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:2.0),
            collect(1.0:3.0), collect(1.0:3.0))

        @test_throws DimensionMismatch f(collect(1.0:2.0), collect(1.0:2.0),
            collect(1.0:3.0), 1.2)

        @test f(collect(1.0:100.0),ones(Float64,100),1.0,1.0) == 5050.0

        @test f(collect(1.0:100.0),2.0*ones(Float64,100),1.0,1.0) == 10100.0

        @test f(1.0:100.0, 1.0:100.0, 1.0:100.0, ones(Float64, 100)) == 100.0

        @test f(1.0:100.0, 1.0:100.0, 1.0:100.0, 1.0) == 100.0

        @test f(1.0:100.0, 1.0:100.0, 1.0, 1.0 ./ (1.0:100.0)) == 5050.0

        @test f(Float64[], Float64[], Float64[], Float64[]) == 0.0

        @test f(Float64[], Float64[], Float64[], 10.0) == 0.0

        @test f(Float64[], Float64[], 2.5, Float64[]) == 0.0

        @test f(Float64[], Float64[], 2.0, 3.0) == 0.0

        samplevalueindices = vcat(trues(100), falses(100))

        @test_throws BoundsError f(samplevalueindices, 1.0:100.0, 1.0:100.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:100.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:100.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:200.0, 1.0:100.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:300.0, 1.0:300.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:300.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:300.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:200.0, 1.0:300.0)

        @test f(samplevalueindices, 1.0:200.0, ones(Float64, 200), 1.0, 1.0) ==
            5050.0

        @test f(samplevalueindices, 1.0:200.0, ones(Float64, 200),
            ones(Float64,200), 1.0) == 5050.0

        @test f(samplevalueindices, 1.0:200.0, ones(Float64, 200), 1.0,
            ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, 1.0:200.0, ones(Float64, 200),
            ones(Float64, 200), ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, ones(Float64, 200), 1.0:200.0,
            ones(Float64, 200), ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, ones(Float64, 200), 1.0:200.0, 1.0, 1.0) ==
            5050.0

        @test f(samplevalueindices, ones(Float64, 200), 1.0:200.0,
            ones(Float64,200), 1.0) == 5050.0

        @test f(samplevalueindices, ones(Float64, 200), 1.0:200.0, 1.0,
            ones(Float64,200)) == 5050.0

        f = BayesianNonparametricStatistics.calculateΔt

        @test f(0.0:0.1:100.0) == 0.1

        @test f(0.0:0.1:-0.1) == 0.1

        @test f([0.0, 1.0, 100.0]) == [1.0, 99.0]

        f = BayesianNonparametricStatistics.calculategirsanovmatrix

        # bla.

        f = BayesianNonparametricStatistics.calculategirsanovvectorelement

        @test f(1.0:100.0, ones(Float64, 100), ones(Float64,100)) == 5050.0

        @test f(1.0:100.0, ones(Float64, 100), 1.0) == 5050.0

        @test_throws DimensionMismatch f(1.0:100.0, 1.0:50.0, 2.0)

        @test_throws DimensionMismatch f(1.0:66.0, 1.0:100.0, 2.0)

        @test_throws DimensionMismatch f(1.0:100.0, 1.0:50.0, 1.0:50.0)

        @test_throws DimensionMismatch f(1.0:50.0, 1.0:100.0, 1.0:50.0)

        @test_throws DimensionMismatch f(1.0:100.0, 1.0:100.0, 1.0:50.0)

        @test f(1.0:10.0, 1.0:10.0, 0.0) == Inf

        @test f(1.0:2.0, 1.0:2.0, [0.0, 1.0]) == Inf

        samplevalueindices = vcat(trues(100), falses(100))

        @test f(samplevalueindices, collect(1.0:200.0), collect(1.0:200.0).^2, 1.0) == sum(k^3
            for k in 1:100)

        @test f(samplevalueindices, collect(1.0:200.0), collect(1.0:200.0).^2, collect(1.0/k
            for k in 1.0:200.0)) == sum(k^5 for k in 1:100)

        @test_throws BoundsError f(samplevalueindices, 1.0:100.0, 1.0:100.0,
            1.0)

        @test_throws BoundsError f(samplevalueindices, 1.0:100.0, 1.0:100.0,
            1.0:100.0)

        f = BayesianNonparametricStatistics.calculategirsanovvector

        # lengthvector = 2
        samplevalues = [1.0, 2.0]
        ψXt = [[1.0], [1.0]]
        σXt = [1.0]
        @test f(2, samplevalues, ψXt, σXt) == [1.0, 1.0]

        # lengthvector = 2
        samplevalues = [1.0, 2.0]
        ψXt = [[1.0], [1.0]]
        σXt = 1.0
        @test f(2, samplevalues, ψXt, σXt) == [1.0, 1.0]

        samplevalueindices = [BitArray([true, false]), BitArray([false, true])]

        # lengthvector = 2
        samplevalues = [1.0, 2.0, 3.0]
        ψXt = [[1.0, 1.0], [1.0, 1.0]]
        σXt = [1.0, 1.0]
        @test f(2, samplevalueindices, samplevalues, ψXt, σXt) == [1.0, 1.0]

        # lengthvector = 2
        samplevalues = [1.0, 2.0, 3.0]
        ψXt = [[1.0, 1.0], [1.0, 1.0]]
        σXt = 1.0
        @test f(2, samplevalueindices, samplevalues, ψXt, σXt) == [1.0, 1.0]


        f = BayesianNonparametricStatistics.calculateσXt

        @test f(1.0, 1.0:10.0) == 1.0

        @test f(identity, 1.0:10.0) == collect(1.0:10.0)

        α = 0.5
        Π = FaberSchauderExpansionWithGaussianCoefficients([2^(α*j) for j in 1:5])

        model = SDEModel(1.0, 0.0, 10000.0, 0.01)
        sde = SDE(x->0.0, model)
        X = rand(sde)


        postΠ = calculateposterior(Π, X, model)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(undef, length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end
        @test maximum(abs.(StatsBase.mean(x, dims = 2))) < 0.1

        Π = GaussianProcess([fourier(k) for k in 1:40], GaussianVector(sparse(Diagonal([k^(-1.0) for k in 1:40]))))

        model = SDEModel(1.0, 0.0, 10000.0, 0.01)
        sde = SDE(x->0.0, model)
        X = rand(sde)

        postΠ = calculateposterior(Π, X, model)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(undef, length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end

        @test maximum(abs.(StatsBase.mean(x, dims = 2))) < 0.1
    end # testset calculateposterior.jl
end # testset. 