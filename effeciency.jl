using Yao
using Plots
using LinearAlgebra
using Random
using Statistics

rng = Random.GLOBAL_RNG

function testfidelity(Σ,reg2,k::Int,ϵ)
    Nt = 200*k/(ϵ^2)
    aΣ = [sin(ϵ*Σ) cos(ϵ*Σ); -cos(ϵ*Σ) sin(ϵ*Σ)]
    mas = matblock(aΣ)
    ms = matblock(Σ)
    regt = copy(reg2)
    reg = join(zero_state(1),reg2)
    for i in k
        regt |> ms |> normalize!
    end
    m = Measure(3,locs=(3,))
    num = 0
    measuretimes = 0
    for i in 1:Nt
        if num == k
            break
        else
            reg |> mas |> m
            if m.results[1] == 0
                num +=1
            end
            reg |> Measure{3,1,AbstractBlock,typeof(rng)}(rng,Z,(3,),0,false)
            measuretimes += 1
        end
    end
    f0 = (fidelity(reg,join(zero_state(1),regt)))[1]
    if num == k
        return measuretimes,f0
    else
        return measuretimes, false
    end
end

function onefp(m::Int = 10,n::Int = 50,dis::Float64 = 0.05)
    p = zeros(m,n)
    f = zeros(m,n)
    for i in 1:m
        Σ = Diagonal(rand(4))
        for j in 1:n
            ft,pt = nonunitfidprob(Σ,dis*j)
            p[i,j] = pt
            f[i,j] = ft[1]
        end
    end
    ap = sum(p,dims=1)./m
    af = sum(f,dims=1)./m
    labels = ["probability of 0" "fidelity"]
    plot(dis*(1:n),[vec(ap) vec(af)],label=labels,title="Single non-unitary gate")
    xlabel!("epsilon")
end

function multifp(Σ::Diagonal,k::Int,ϵ)
    Nt = 5000
    u = matblock([sin(ϵ*Σ) cos(ϵ*Σ);-cos(ϵ*Σ) sin(ϵ*Σ)])
    reg2 = rand_state(2)
    regt = copy(reg2)
    for i in 1:k
        regt |> matblock(Σ) |> normalize!
    end
    ancilla = zero_state(1)
    reg = join(ancilla,reg2)
    m = Measure(3,locs=(3,))
    m0 = Measure{3,1,AbstractBlock,typeof(rng)}(rng,Z,(3,),0,false)
    num = 0
    ni = 1
    for i in 1:Nt
        if num == k
            break
        end
        reg |> u |> m
        if m.results[1] == 0
            num = num + 1
        end
        reg |> m0
        ni +=1
    end
    if num == k
        f = fidelity(reg,join(zero_state(1),regt))
        return num,ni, f
    else
        return num
    end
end

function plotmulti(Σ::Diagonal,kmax::Int,nsample::Int,ϵ)
    knf = zeros(kmax-1,nsample,2)
    for i in 1:kmax-1
        for j in 1:nsample
            num,ni,f = multifp(Σ,i+1,ϵ)
            knf[i,j,1] = ni
            knf[i,j,2] = f[1]
        end
    end
    measuretimes = knf[:,:,1]
    measurefidelity = knf[:,:,2]
    nt = sum(measuretimes,dims = 2)./nsample
    nf = sum(measurefidelity,dims = 2)./nsample
    plot(2:kmax,vec(nt),title="Average measure times")
    savefig()
    plot(2:kmax,vec(nf),title="Average fidelity")
    savefig("D://Nonunitary//averagefidelity")
end

function measurepsilon(nummat::Int,k::Int,nsample::Int,rangepsilon)
    nt = zeros(nummat,size(rangepsilon)[1])
    nf = zeros(nummat,size(rangepsilon)[1])
    for t in 1:nummat
        Σ = Diagonal(rand(4))
        trip = zeros(size(rangepsilon)[1],nsample,2)
        i = 0
        for ϵ in rangepsilon
            i += 1
            for j in 1:nsample
                num,ni,f = multifp(Σ,k,ϵ)
                trip[i,j,1] = ni
                trip[i,j,2] = f[1]
            end
        end
        measuretimes = trip[:,:,1]
        measurefidelity = trip[:,:,2]
        nt[t,:] = sum(measuretimes,dims = 2)./nsample
        nf[t,:] = sum(measurefidelity,dims = 2)./nsample
    end
    avernt = sum(nt,dims = 1)./nummat
    avernf = sum(nf,dims = 1)./nummat
    plot(rangepsilon,vec(avernt),xlabel="epsilon",ylabel="average measure times",title="k = $k")
    savefig("D://Nonunitary//constkaveragetimes")
    plot(rangepsilon,vec(avernf),xlabel="epsilon",ylabel="average fidelity",title="k = $k")
    savefig("D://Nonunitary//constkaveragefidelity")
end

function plotfidelity(Σ,reg2,nummat::Int,numk::Int,ϵ)
    f = Float64[]
    m = Int[]
    count = zeros(5)
    results = [testfidelity(Σ,reg2,numk,ϵ) for i in 1:nummat]
    for i in results
        if i[2] != false
            push!(f,i[2])
            push!(m,i[1])
        end
    end
    scatter(1:length(f),f,title="Distribution of fidelity after 50 samples (success times :$(length(f)))")
    savefig("D://Nonunitary//fidelityscatter_$numk")
    scatter(1:length(m),m,title="Distribution of measuretimes after 50 samples (success times :$(length(m)))")
    savefig("D://Nonunitary//measuretimescatter_$numk")
    histogram(f,bins=0.0:0.05:1.0)
    title!("Distribution of fidelity after 50 samples (success times :$(length(f)))")
    savefig("D://Nonunitary//fidelityhis_$numk")
    #=histogram(m,bins=numk:50:)=#
end

function doublepsilon(ϵ::Float64)
    return sqrt(1-sqrt(1-ϵ^2))
end

function testpoly(Σ,reg,nsample::Int,lengthmax::Int)
    f = Float64[]
    m = Int64[]
    meas = Float64[]
    fid = Float64[]
    for i in 1:lengthmax
        ϵ = 0.1/sqrt(i)
        results = [testfidelity(Σ,reg,i,ϵ) for j in 1:nsample]
        for r in results
            if r[2]!=false
                push!(f,r[2])
                push!(m,r[1])
            end
        end
        push!(fid,mean(f))
        push!(meas,mean(m))
    end
    fid,meas
end

Σ = Diagonal(Complex.(rand(4)))
reg = rand_state(2)

testpoly(Σ,reg,50,5)
