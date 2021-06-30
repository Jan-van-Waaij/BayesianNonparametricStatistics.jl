using Test, Random, InteractiveUtils, SparseArrays, LinearAlgebra, StatsBase, Distributions

using BayesianNonparametricStatistics

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

        σ2(x) = 2+sin(x)
        M = SDEModel(σ2, 1.3, 101.0, 0.5)
        @test M.σ == σ2
        @test M.beginvalue == 1.3
        @test M.endtime == 101.0
        @test M.Δ == 0.5

        @test_throws ArgumentError SDEModel(sin, 0.0, -1.0, 0.1)
        @test_throws ArgumentError SDEModel(sin, 0.0, 1.0, -0.1)
        @test_throws ArgumentError SDEModel(sin, 0.0, -1.0, -0.1)
        @test_throws ArgumentError SDEModel(1.0, 0.0, 0.0, 0.1)
        @test_throws ArgumentError SDEModel(1.0, 0.0, 10.0, 0.0)
        @test_throws ArgumentError SDEModel(1.0, 0.0, 0.0, 0.0)
    end # perfect

    #Test functions are correct, complete and the same in both versions. 
    @testset "samplepath.jl" begin
        @test supertype(SamplePath) == AbstractSamplePath

        @test supertype(AbstractSamplePath) == Any

        @test BayesianNonparametricStatistics.isstrictlyincreasing(0:0.1:1)
        @test !BayesianNonparametricStatistics.isstrictlyincreasing(0:-0.1:-1.0)
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([0.0, 0.1, 0.2, 0.1, 0.3, 0.4])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([0.0, -0.1, 0.0, 0.2, 0.3])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([1, 2, 3, 4, -10, 11, 12])
        @test BayesianNonparametricStatistics.isstrictlyincreasing([1, 2, 3, 4, 500, 600, 2000, 10000])
        @test BayesianNonparametricStatistics.isstrictlyincreasing([1.0, 2.0, 2.00000001, 2.00000002])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([2.0,2.0,2.0,2.0,2.0])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([2.0,2.0,2.1,2.2,2.3])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([2.0,2.1,2.2,2.2,2.3])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([2.1,2.2,2.3,2.4,2.4])
        @test !BayesianNonparametricStatistics.isstrictlyincreasing([2.0,2.0, 3.0, 3.0001])

        X = SamplePath(0.0:0.1:1.0, sin)
        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == sin.(0.0:0.1:1.0)

        X = SamplePath(0.0:0.1:1.0, 1.0:11.0)

        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == 1.0:11.0

        X = SamplePath(0.0:0.1:1.0, x->2*x)
        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == 0.0:0.2:2.0

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
        X.timeinterval == -2.0:0.01:2.0
        X.samplevalues == sin.(-2.0:0.01:2.0)

        X = SamplePath(-2.0:0.01:2.0, sin)
        X.timeinterval == -2.0:0.01:2.0
        X.samplevalues == sin.(-2.0:0.01:2.0)

        X = SamplePath(0.0:0.1:2π, sin)
        @test length(X) == length(0.0:0.1:2π)

        @test_throws DimensionMismatch SamplePath(0.0:2.0, [1.0, 2.0])
        @test_throws DimensionMismatch SamplePath(0.0:2.0, [1.0, 2.0, 4.0, 5.0])

        @test_throws ArgumentError SamplePath(0.0:-0.1:-0.1, [0.0, 2.0])

        @test_throws DimensionMismatch SamplePath([0.1, 2.0],[3.0])
        @test_throws DimensionMismatch SamplePath([0.1, 2.0],Float64[])
        @test_throws DimensionMismatch SamplePath(Float64[],[3.0])

        @test_throws ArgumentError SamplePath([0.1, 2.0, 1.0], [1.0,2.0,3.0])
        @test_throws ArgumentError SamplePath(0.0:-1.0:-10.0, rand(11))

        @test step(SamplePath(0.0:0.2:1.0, sin)) == 0.2
        @test step(SamplePath(0.0:10.0:1000.0, 0.0:100.0)) == 10.0

        for i in 1:10 
            @test length(SamplePath(1.0:i, rand(i))) == i 
        end 

        @test length(SamplePath([1.0, 2.0, 4.0, 5.0, 16.0, 30.0, 34.0, 100.0], x->x^2)) == 8
        @test length(SamplePath([1.0, 2.0, 4.0, 5.0, 16.0, 30.0, 34.0, 100.0], [90.0, 56.0, 49.0, 25.0, 6.0, 10.0, 19.0, 1000000.0])) == 8        

        t = 0.0:0.1:2π
        X = SamplePath(t, sin)
        @test [value for value in X] == X.samplevalues
        @test try [value for value in X]
            true
        catch
            false
        end
        @test sum(X) ≈ sum(X.samplevalues)
        @test [value for value in Iterators.Reverse(X)] == reverse(X.samplevalues)
        @test sum(Iterators.Reverse(X)) ≈ sum(X) 
        @test maximum(X) == maximum(X.samplevalues)
        @test minimum(X) == minimum(X.samplevalues)
        @test extrema(X) == extrema(X.samplevalues)

        @test firstindex(X) == 1 
        @test lastindex(X) == length(X.samplevalues)

        Y = SamplePath([0.1, 0.2], [1.0, 2.0])
        @test eltype(typeof(X)) == Float64
        @test eltype(typeof(Y)) == Float64

        @test X[begin] == sin(0.0)
        @test X[end] == sin(t[end])

        X = SamplePath(0:100.0, 0:100.0)
        @test sum(X) == 5050
        @test X[begin] == 0.0
        @test X[end] == 100.0
        @test [value for value in Iterators.Reverse(X)][1] == 100.0
    end # Perfect

    #Test functions are correct, complete and the same in both versions. 
    @testset "basisfunctions.jl" begin
        numberofbasisfunctionstested = 200
        numberoffaberschauderlevelstested = convert(Int64, round(log2(numberofbasisfunctionstested), RoundUp)) - 1
        numberoffourierfunctionstested = numberofbasisfunctionstested
        
        # Exceptions

        @test_throws AssertionError fourier(-1)
        @test_throws AssertionError fourier(-10)

        #The Fourier functions are orthogonal.
        t = 0.0:1/(4*ceil(numberoffourierfunctionstested/2)):1.0
        tprecise = 0.0:10.0^-1/(4*ceil(numberoffourierfunctionstested/2)):1.0
        for k in 0:numberoffourierfunctionstested-1
            for l in k+1:numberoffourierfunctionstested
                phikvalues = fourier(k).(tprecise[1:end-1])
                philvalues = fourier(l).(tprecise[1:end-1])
                @test abs(sum(step(tprecise)*phikvalues.*philvalues)) < 10.0^-10
            end
        end
        # And the square integrates to one.
        for k in 0:numberoffourierfunctionstested
            values = fourier(k).(tprecise)
            @test abs(step(tprecise)*sum(values.*values)-1.0) < 10.0^-2
        end

        # They are one-periodic.
        for k in 0:numberoffourierfunctionstested
            for y in t
                value = fourier(k)(y)
                for m in -10.0:10.0
                    valuetwo = fourier(k)(y+m)
                    @test abs(value - valuetwo) < 10.0^-10
                end
            end
        end

        # fourier(0) is constant 1.0.
        for x in -10.0:0.01:10.0
            @test fourier(0)(x) == 1.0
        end 
        # # For k≥1 they have minimum -sqrt(2) and maximum sqrt(2).
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
        # # For k≥1:
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
        
        # tests for Faber-Schauder functions
        #
        @test_throws AssertionError faberschauder(-1, 1)
        @test_throws AssertionError faberschauder(-3, 1)
        @test_throws AssertionError faberschauder(1, 0)
        @test_throws AssertionError faberschauder(1, -2)
        @test_throws AssertionError faberschauder(1, 3)

        for j in 0:numberoffaberschauderlevelstested
            if j>0 
                @test_throws AssertionError faberschauder(-j, 1)
            end
            @test_throws AssertionError faberschauder(j, 0)
            @test_throws AssertionError faberschauder(j, -1)
            @test_throws AssertionError faberschauder(j, 2^j + 1)
            @test_throws AssertionError faberschauder(j, 2^j + 2)
        end 

        # # Below we test Faber-Schauder functions on their mathematical properities, up
        # # to level numberoffaberschauderlevelstested, defined above.
        t = 0.:2.0^(-numberoffaberschauderlevelstested-2):1.0
        
        # Orthogonality tests

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

        # ψ_{j,k} is independent with ψ_{j+r, 1},..., ψ_{j+r, (k-1)*2^r} and with
        # ψ_{j+r, k*2^r+1},...,ψ_{j+r, 2^(j+r)}
        for j in 1:numberoffaberschauderlevelstested
            for k in 1:2^j
                for r in 0:(numberoffaberschauderlevelstested-j)
                    for ℓ in 1:(k-1)*2^r 
                        psijkvalues = faberschauder(j,k).(t[1:end-1])
                        psijplusrlvalues = faberschauder(j+r,ℓ).(t[1:end-1])
                        @test abs(step(t)*sum(psijkvalues.*psijplusrlvalues))<10.0^-3
                    end 
                    for ℓ in k*2^r + 1:2^(j+r)
                        psijkvalues = faberschauder(j,k).(t[1:end-1])
                        psijplusrlvalues = faberschauder(j+r,ℓ).(t[1:end-1])
                        @test abs(step(t)*sum(psijkvalues.*psijplusrlvalues))<10.0^-3
                    end 
                end 
            end 
        end 

        # Integral 

        t = 0.:2.0^(-numberoffaberschauderlevelstested-2):1.0
        # They integrate to 2^(-j-1)
        psionevalues = faberschauderone.(t[1:end-1])
        @test abs(step(t)*sum(psionevalues)-2.0^-1) < 10.0^-3
        
        for j in 1:numberoffaberschauderlevelstested
            for k in 1:2^j
                psijkvalues = faberschauder(j,k).(t[1:end-1])
                @test abs(step(t)*sum(psijkvalues)-2.0^(-j-1)) < 10.0^-3
            end
        end

        # Maxima and minima.

        t = 0.:2.0^(-numberoffaberschauderlevelstested-3):1.0
        # They have global minimum 0.0 and global maximum 1.0
        @test maximum(faberschauderone.(t)) == 1.0
        @test minimum(faberschauderone.(t)) == 0.0

        # psi_1 has maximum 1 at 0 and 1.
        @test faberschauderone(0.0) == 1.0
        @test faberschauderone(1.0) == 1.0

        # psi_1 has minimum 0 at 0.5.
        @test faberschauderone(0.5) == 0.0

        # Also ψ_{j,k} has global minimum 0 and global maximum 1.
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                values = faberschauder(j,k).(t)
                @test minimum(values) == 0.0
                @test maximum(values) == 1.0
            end
        end

        # psi_{j,k} has maximum 1.0 at 2.0^(-j-1)+(k-1)*2.0^-j
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                @test faberschauder(j,k)(2.0^(-j-1)+(k-1)*2.0^-j) == 1.0
            end
        end

        #psi_{j,k} has minimum 0.0 at (k-1)*2.0^-j and k*2.0^-j
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                @test faberschauder(j,k)((k-1)*2.0^-j) == 0.0
                @test faberschauder(j,k)(k*2.0^-j) == 0.0
            end
        end

        # Values at certain places. 

        #psi_{j,k} are zero for 0.0≤x≤(k-1)*2.0^-j and k*2.0^-j ≤x≤1.0
        x = 0.:2.0^(-numberoffaberschauderlevelstested-3):1.0
        for j in 1:numberoffaberschauderlevelstested
            for k in 1:2^j
                t_left = 0.:2.0^(-numberoffaberschauderlevelstested-3):(k-1)*2.0^-j
                t_right = k*2.0^-j:2.0^(-numberoffaberschauderlevelstested-3):1.0
                ψjk = faberschauder(j,k)
                @test all(ψjk.(t_left) .== 0.0)
                @test all(ψjk.(t_right) .== 0.0)  
            end
        end

        # ψ_1 is 1/2 at 0.25 and 0.75
        # ψ_{j,k} is 1/2 at (k-1+1/4)*2.0^-j and at (k-1+3/4)*2^-j.
        @test faberschauderone(0.25) == 1/2
        @test faberschauderone(0.75) == 1/2
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j 
                ψjk = faberschauder(j,k)
                @test ψjk((k-1+1/4)*2.0^-j) == 1/2
                @test ψjk((k-1+3/4)*2.0^-j) == 1/2
            end 
        end 

        # ψ_1 is 1/4 at 3/8 and 5/8
        # ψ_{j,k} is 1/4 at (k-1+1/8)*2.0^-j and at (k-1+7/8)*2^-j.
        @test faberschauderone(3/8) == 1/4
        @test faberschauderone(5/8) == 1/4
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j 
                ψjk = faberschauder(j,k)
                @test ψjk((k-1+1/8)*2.0^-j) == 1/4
                @test ψjk((k-1+7/8)*2.0^-j) == 1/4
            end 
        end 

        # ψ_1 is 3/4 at 1/8 and 7/8
        # ψ_{j,k} is 1/4 at (k-1+3/8)*2.0^-j and at (k-1+5/8)*2^-j.
        @test faberschauderone(1/8) == 3/4
        @test faberschauderone(7/8) == 3/4
        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j 
                ψjk = faberschauder(j,k)
                @test ψjk((k-1+3/8)*2.0^-j) == 3/4
                @test ψjk((k-1+5/8)*2.0^-j) == 3/4
            end 
        end 

        # Periodicity

        # They are one-periodic.
        x = 0.:2.0^(-numberoffaberschauderlevelstested-3):1.0
        for y in x
            value = faberschauderone(y)
            for k in -10.0:10.0
                valuetwo = faberschauderone(y+k)
                @test value == valuetwo
            end
        end

        for j in 0:numberoffaberschauderlevelstested
            for k in 1:2^j
                for y in x
                    value = faberschauder(j,k)(y)
                    for m in -10.:10.
                        valuetwo = faberschauder(j,k)(y+m)
                        @test value == valuetwo
                    end
                end
            end
        end

        # End of tests for mathematical properities of Faber-Schauder functions.
    end # Perfect.

    #Test functions are correct, complete and the same in both versions. 
    @testset "SDE.jl" begin
        @test supertype(SDE) == AbstractSDE

        @test supertype(AbstractSDE) == Any

        @test SDE in subtypes(AbstractSDE)

        model = SDEModel(1.0, 0.0, 10.0, 0.001)
        sde = SDE(sin, model)
        @test sde.b == sin 
        @test sde.model.beginvalue == 0.0
        @test sde.model.σ == 1.0
        @test sde.model.endtime = 10.0
        @test sde.model.Δ == 0.001

        model = SDEModel(abs, -2.0, 1098.0, 0.02)
        sde = SDE(cos, model)
        @test sde.b == cos 
        @test sde.model.beginvalue == -2.0
        @test sde.model.σ == abs 
        @test sde.model.endtime = 1098.0
        @test sde.model.Δ == 0.02

        # The following should represent a Brownian motion.
        model = SDEModel(1.0, 0.0, 1.0, 0.001)
        sde = SDE(x->0.0, model)

        lengthvector = 1000
        x = Vector{Float64}(undef, lengthvector)
        
        for k in 1:lengthvector
            X = rand(sde)
            x[k] = X[end]
        end

        @test abs(StatsBase.mean(x)) < 0.1
        @test 0.9 < StatsBase.var(x) < 1.1

        X = rand(sde)

        @test length(X) == length(0.0:sde.model.Δ:sde.model.endtime)
        @test X.timeinterval == 0.0:sde.model.Δ:sde.model.endtime

        model = SDEModel(identity, 1.0, 1.0, 0.001)
        geometricbrownianmotion = SDE(x->0.5*x, model)

        lengthvector = 1000_000
        x = Vector{Float64}(undef, lengthvector)
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

        f = BayesianNonparametricStatistics.calculatenextsamplevalue
        σ = 1.0 
        Δ = 1.0
        prevXval = 1.0
        b = identity
        BMincrement = 1.0
        @test f(prevXval, σ, b, Δ, BMincrement) == 3.0

        σ = 1.0 
        Δ = 1.0
        prevXval = 1.0
        b = x->1.0
        BMincrement = 1.0
        @test f(prevXval, σ, b, Δ, BMincrement) == 3.0
        

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


        d_distributions = MvNormal([1.0,2.0, 3.0, 4.0])

        @test_throws DimensionMismatch GaussianProcess([fourier(k) for k in 1:2], GaussianVector(sparse(Diagonal([1.0,1.0,1.0]))))

        @test_throws DimensionMismatch GaussianProcess([fourier(k) for k in 1:2], d_distributions)

        @test_throws DimensionMismatch GaussianProcess([fourier(k) for k in 1:4], GaussianVector(sparse(Diagonal([1.0,1.0,1.0]))))

        @test_throws DimensionMismatch GaussianProcess([fourier(k) for k in 1:5], d_distributions)

        X = GaussianProcess([sinpi, cospi], GaussianVector(sparse(Diagonal([1.0, 1.0]))))

        X_distributions = GaussianProcess([sinpi, cospi], MvNormal([1.0, 2.0]))
        
        @test X.basis == [sinpi, cospi]
        @test X_distributions.basis == [sinpi, cospi]
        @test typeof(X.distribution) <: GaussianVector
        @test typeof(X_distributions.distribution) <: AbstractMvNormal
        @test X.distribution.Σ == [1.0 0.0; 0.0 1.0]
        @test cov(X_distributions.distribution) == [1.0 0.0; 0.0 4.0]
        @test mean(X.distribution) == [0.0,0.0]
        @test mean(X_distributions.distribution) == [0.0, 0.0] 
        @test length(X) == 2
        @test length(X_distributions) == 2 
        @test typeof(rand(X)) <: Function     
        @test typeof(rand(X_distributions)) <: Function
        
        n = 100000
        a = sinpi(1/4)
        x = Vector{Float64}(undef, n) 
        for k in 1:n
            f = rand(X)
            x[k] = f(1/4)
        end

        @test abs(StatsBase.mean(x))<0.01
        @test abs(StatsBase.var(x)-2*a^2)<0.1

        X_distributions = GaussianProcess([sinpi, cospi], MvNormal([1.0,1.0]))
        n = 100000
        a = sinpi(1/4)
        x = Vector{Float64}(undef, n) 
        for k in 1:n
            f = rand(X_distributions)
            x[k] = f(1/4)
        end

        @test abs(StatsBase.mean(x))<0.01
        @test abs(StatsBase.var(x)-2*a^2)<0.1

        x = 0.0:0.1:1.0
        y = BayesianNonparametricStatistics.sumoffunctions([sin, cos], [1., 1.]).(x)
        z = map(x-> sin(x)+cos(x), x)
        @test y ≈ z

        A = sparse(Diagonal([1.0, 1.0]))

        distribution = GaussianVector(A)

        @test_throws AssertionError FaberSchauderExpansionWithGaussianCoefficients(1,distribution)

        @test_throws AssertionError FaberSchauderExpansionWithGaussianCoefficients(Vector{Float64}(undef, 0))

        Π = FaberSchauderExpansionWithGaussianCoefficients(0,distribution)
        @test length(Π) == 2

        f = BayesianNonparametricStatistics.calculateboundssupport

        for higestlevel in 1:10
            leftbounds, rightbounds = f(higestlevel)
            leftboundsnext, rightboundsnext = f(higestlevel+1)
            @test length(leftbounds) == 2^(higestlevel+1)
            @test length(rightbounds) == 2^(higestlevel+1)
            @test leftbounds[1] == 0.0
            @test rightbounds[1] == 1.0
            @test leftboundsnext[1:length(leftbounds)] == leftbounds
            @test rightboundsnext[1:length(rightbounds)] == rightbounds
            @test leftbounds[end-2^higestlevel+1] == 0.0
            @test rightbounds[end] == 1.0
            for k in 1:2^higestlevel
                @test leftbounds[end - 2^higestlevel + k] == (k-1)*2.0^(-higestlevel)
                @test rightbounds[end - 2^higestlevel + k] == k*2.0^(-higestlevel) 
            end 
        end 

        f = BayesianNonparametricStatistics.createFaberSchauderBasisUpToLevelHigestLevel

        for i in 1:10 
            v = f(i)
            vnext = f(i+1)
            @test length(v) == 2^(i+1)
            @test v[1] == faberschauderone
            @test vnext[1:length(v)] == v 
            @test length(unique(v)) == length(v)
            @test try [item(0.5) for item in v]
                true
            catch
                false
            end
        end 

        f = BayesianNonparametricStatistics.createvectorofstandarddeviationsfromstandarddeviationsperlevel

        standarddeviationsperlevel = 1.0:4.0

        @test f(standarddeviationsperlevel) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

        @test_throws AssertionError f(Float64[])

        for k in 1:10 
            vectorofstandarddeviations = f(1.0:k)
            @test length(vectorofstandarddeviations) == 2^k
            @test vectorofstandarddeviations[1] == 1.0
            @test vectorofstandarddeviations[2] == 1.0
            @test vectorofstandarddeviations[end] == k 
            for ℓ in 1:k 
                @test all(vectorofstandarddeviations[1+2^(ℓ-1):2^ℓ] .== ℓ)
            end 
        end 

        @test_throws AssertionError BayesianNonparametricStatistics.sumoffunctions([sin], [1.0,2.0])

        vectoroffunctions = [sin, x -> cos(x+π/2)]
        vectorofscalars = [1.0, 1.0]
        f = BayesianNonparametricStatistics.sumoffunctions(vectoroffunctions, vectorofscalars)
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

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:100.0, 1.0:100.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:100.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:100.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:200.0, 1.0:100.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:300.0, 1.0:300.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:300.0,
            1.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:300.0, 1.0)

        @test_throws BoundsError f(samplevalueindices, samplevalueindices, 1.0:200.0, 1.0:200.0,
            1.0:200.0, 1.0:300.0)

        @test f(samplevalueindices, samplevalueindices, 1.0:200.0, ones(Float64, 200), 1.0, 1.0) ==
            5050.0

        @test f(samplevalueindices, samplevalueindices, 1.0:200.0, ones(Float64, 200),
            ones(Float64,200), 1.0) == 5050.0

        @test f(samplevalueindices, samplevalueindices, 1.0:200.0, ones(Float64, 200), 1.0,
            ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, samplevalueindices, 1.0:200.0, ones(Float64, 200),
            ones(Float64, 200), ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, samplevalueindices, ones(Float64, 200), 1.0:200.0,
            ones(Float64, 200), ones(Float64,200)) == 5050.0

        @test f(samplevalueindices, samplevalueindices, ones(Float64, 200), 1.0:200.0, 1.0, 1.0) ==
            5050.0

        @test f(samplevalueindices, samplevalueindices, ones(Float64, 200), 1.0:200.0,
            ones(Float64,200), 1.0) == 5050.0

        @test f(samplevalueindices, samplevalueindices, ones(Float64, 200), 1.0:200.0, 1.0,
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
        @test maximum(abs.(mean(postΠ).(y))) < 0.1

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
        @test maximum(abs.(mean(postΠ).(y))) < 0.1
    end # testset calculateposterior.jl
end # testset. 