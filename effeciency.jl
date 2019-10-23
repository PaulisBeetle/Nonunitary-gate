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
    for i in 1:k
        regt |> ms |> normalize!
    end
    m = Measure(3,locs=(3,),collapseto=bit"0")
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

function fidmes(Σ,reg,k::Int,ϵ::Float64)
    x = statevec(reg)
    diff = (x'*Σ^(2k+4)*x)*(x'*Σ^(2k)*x)-(x'*Σ^(2k+2)*x)^2
    A = (x'*Σ^(2k)*x)^2
    approx = [sqrt(abs(1-(n/2-k/3)^2*diff*ϵ^4/A)) for n in k:2000]
    #rstate = Σ^k*x./sqrt((x'*Σ^(2k)*x))
    #exact = [abs(rstate'*sin(ϵ*Σ)^(k)*cos(ϵ*Σ)^(n-k)*x/(sqrt(x'*sin(ϵ*Σ)^(2k)*cos(ϵ*Σ)^(2n-2k)*x)))  for n in k:2000]
end

function plotfidelity(Σ,reg2,nummat::Int,numk::Int,ϵ)
    f = Float64[]
    m = Int[]
    results = [testfidelity(Σ,reg2,numk,ϵ) for i in 1:nummat]
    for i in results
        if i[2] != false
            push!(f,i[2])
            push!(m,i[1])
        end
    end
    scatter(1:length(f),f,title="Distribution of fidelity after $nummat samples (success times :$(length(f)))")
    savefig("D://Nonunitary//fidelityscatter_$numk")
    scatter(1:length(m),m,title="Distribution of measuretimes after $nummat samples (success times :$(length(m)))")
    savefig("D://Nonunitary//measuretimescatter_$numk")
    histogram(f,bins=0.0:0.05:1.0)
    title!("Distribution of fidelity after $nummat samples (success times :$(length(f)))")
    savefig("D://Nonunitary//fidelityhis_$numk")
    histogram(m,bins=0:100:2000,title="Distribution of measure times after $nummat samples (success times: $(length(f)))")
    savefig("D://Nonunitary//measuretimeshis_$numk")
    scatter(results,title="fidelity-measuretimes, k=$numk, eps=$ϵ",xlabel="measuretimes",ylabel="fidelity")
    plot!(k:2000,fidmes(Σ,reg,numk,ϵ))
    savefig("D://Nonunitary//fidmes_k=$(numk)_eps=$(ϵ)")
    #=histogram(m,bins=numk:50:)=#
end

function doublepsilon(ϵ::Float64)
    return sqrt(1-sqrt(1-ϵ^2))
end

function testpoly(Σ,reg,nsample::Int,lengthmax::Int,ϵ)
    meas = Float64[]
    fid = Float64[]
        results = [testfidelity(Σ,reg,lengthmax,ϵ) for j in 1:nsample]
        for r in results
            if r[2]!=false
                push!(fid,r[2])
                push!(meas,r[1])
            end
        end
        mean(fid),mean(meas)
end

function exactprob(Σ::Diagonal,reg,k::Int,ϵ::Float64)
    x = statevec(reg)
    p = Float64[]
    for i in k:5000
        push!(p,abs(binomial(i-1,k-1)*x'*sin(ϵ*Σ)^(2*k)*cos(ϵ*Σ)^(2*i-2*k)*x))
    end
    p
end

function measuretimes(Σ::Diagonal,reg,k::Int,ϵ)
    x = statevec(reg)
    return abs(k*x'*(sin(ϵ*Σ)^(-2))*x)
end

function verifid(Σ::Diagonal,reg,n::Int,k::Int,ϵ)
    x = statevec(reg)
    y = sin(ϵ*Σ)^(k)*cos(ϵ*Σ)^(n-k)*x./sqrt(x'*sin(ϵ*Σ)^(2k)*cos(ϵ*Σ)^(2n-2k)*x)
    xt = Σ^k*x./sqrt((x'*Σ^(2k)*x))
    for i in 1:k
        reg |> matblock(Σ) |> normalize!
    end
    xr = statevec(reg)
    return abs(y'*xt), abs(y'*xr)
end

Σ = Diagonal(Complex.(rand(4)))
reg = rand_state(2)

k = 3
ϵ = 0.1
plotfidelity(Σ,reg,200,k,ϵ)
plot(k:5000,vec(exactprob(Σ,reg,k,ϵ)),xlabel="measure times",ylabel="probability",title="k=$k eps=$ϵ")
sum(exactprob(Σ,reg,k,ϵ))
