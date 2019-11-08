using Yao
using Plots
using LinearAlgebra
using Random
using Statistics
using QuantumInformation

rng = Random.GLOBAL_RNG
II = [1 0;0 1]
@const_gate ZZ = mat(kron(Z,Z))

function testfidelity(Σ,regn,k::Int,ϵ)
    nbit = Int(log(2,size(statevec(regn))[1]))+1
    Nt = 200*k/(ϵ^2)
    Σ = Complex.(Σ)
    aΣ = Complex.([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])
    mas = matblock(aΣ)
    ms = matblock(Σ)
    regt = copy(regn)
    reg = join(zero_state(1),regn)
    for i in 1:k
        regt |> ms |> normalize!
    end
    m = Measure(nbit,locs=(nbit,),collapseto=bit"0")
    num = 0
    measuretimes = 0
    for i in 1:Nt
        if num == k
            break
        else
            reg |> mas |> m
            if m.results[1] == 1
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
    #approx = [sqrt(abs(1-(n/2-k/3)^2*diff*ϵ^4/A)) for n in k:2000]
    rstate = Σ^k*x./sqrt((x'*Σ^(2k)*x))
    exact = [abs(rstate'*sin(ϵ*Σ)^(k)*cos(ϵ*Σ)^(n-k)*x/(sqrt(x'*sin(ϵ*Σ)^(2k)*cos(ϵ*Σ)^(2n-2k)*x)))  for n in k:2000]
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
    scatter(1:length(f),f,title="Distribution of fidelity after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//fidelityscatter_k$(numk)_eps$(ϵ)")
    scatter(1:length(m),m,title="Distribution of measuretimes after $(length(f)), eps=$ϵ")
    savefig("D://Nonunitary//measuretimescatter_k$(numk)_eps$ϵ")
    histogram(f,bins=0.0:0.05:1.0)
    title!("Distribution of fidelity after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//fidelityhis_$(numk)_eps$ϵ")
    histogram(m,bins=0:100:2000,title="Distribution of measure times after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//measuretimeshis_$(numk)_eps$ϵ")
    scatter(results,title="fidelity-measuretimes, k=$numk, eps=$ϵ",xlabel="measuretimes",ylabel="fidelity")
    plot!(numk:2000,fidmes(Σ,reg2,numk,ϵ))
    savefig("D://Nonunitary//fidmes_k=$(numk)_eps=$(ϵ)")
    f,m
    #=histogram(m,bins=numk:50:)=#
end


function exactprob(Σ::Diagonal,reg,k::Int,ϵ::Float64)
    x = statevec(reg)
    p = Float64[]
    for i in k:50000
        push!(p,abs((binomial(BigInt(i-1),BigInt(k-1))*x'*sin(ϵ*Σ)^(2*k)*cos(ϵ*Σ)^(2*i-2*k)*x)))
    end
    p
end

function measuretimes(Σ::Diagonal,reg,k::Int,ϵ)
    x = statevec(reg)
    return abs(k*x'*(sin(ϵ*Σ)^(-2))*x)
end

function exactfid(Σ::Diagonal,reg,n::Int,k::Int,ϵ)
    x = statevec(reg)
    num = (x'*Σ^k*sin(ϵ*Σ)^k*cos(ϵ*Σ)^(n-k)*x)^2
    den = (x'*Σ^(2k)*x)*(x'*sin(ϵ*Σ)^(2k)*cos(ϵ*Σ)^(2n-2k)*x)
    return abs(num/den)
end

function testfidelity(Σ12,Σ23,reg3,k::Int,ϵ)
    Nt = 50000
    Σ = Diagonal(diag(kron(Σ12,II)*kron(II,Σ23)))
    aΣ = Complex.([cos(ϵ*Σ) -sin(ϵ*Σ);sin(ϵ*Σ) cos(ϵ*Σ)])
    mas = matblock(aΣ)
    ms = matblock(Σ)
    regt = copy(reg3)
    reg = join(zero_state(1),reg3)
    m = Measure(4,locs=(4,),collapseto=bit"0")
    for i in 1:k
        regt |> ms |> normalize!
    end
    num = 0
    measuretimes = 0
    for i in 1:Nt
        if num == k
            break
        else
            reg |> mas |> m
            if m.results[1] == 1
                num += 1
            end
            measuretimes += 1
        end
    end
    f0 = (fidelity(reg,join(zero_state(1),regt)))[1]
    if num == k
        return measuretimes, f0
    else
        return measuretimes, false
    end
end

function plotfidelity(Σ1,Σ2,reg3,nummat::Int,numk::Int,ϵ)
    f = Float64[]
    m = Int[]
    results = [testfidelity(Σ1,Σ2,reg3,numk,ϵ) for in in 1:nummat]
    for i in results
        if i[2] != false
            push!(f,i[2])
            push!(m,i[1])
        end
    end
    scatter(1:length(f),f,title="Distribution of fidelity after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//fidelityscatter_k$(numk)_eps$(ϵ)")
    scatter(1:length(m),m,title="Distribution of measuretimes after $(length(f)), eps=$ϵ")
    savefig("D://Nonunitary//measuretimescatter_k$(numk)_eps$ϵ")
    histogram(f,bins=0.0:0.05:1.0)
    title!("Distribution of fidelity after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//fidelityhis_$(numk)_eps$ϵ")
    histogram(m,bins=0:100:2000,title="Distribution of measure times after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//measuretimeshis_$(numk)_eps$ϵ")
    scatter(results,title="fidelity-measuretimes, k=$numk, eps=$ϵ",xlabel="measuretimes",ylabel="fidelity")
    savefig("D://Nonunitary//fidmes_k=$(numk)_eps=$(ϵ)")
    f,m
end

function threshold(percent::Float64,Σ,reg,k::Int,ϵ,maxn)
    function f(x)
        return exactfid(Σ,reg,x,k,ϵ)-percent
    end
    maxifid = exactfid(Σ,reg,k,k,ϵ)
    for i in k:maxn
        if f(i) < 0
            return i-1
        end
        if i == maxn
            return false
        end
    end
end

function percentcount(percent::Float64,f)
    num = Int(0)
    for i in f
        if i > 0.9
            num += 1
        end
    end
    num
end

function partion(percent::Float64,Σ,reg,k::Int,ϵ,maxn)
    x = statevec(reg)
    sum(i->abs(x*binomial(BigInt(i-1),BigInt(k-1))*sin(ϵ*Σ)^(2k)*cos(ϵ*Σ)^(2n-2k)*x),k:threshold(percent,Σ,reg,k,ϵ))
end


function assume(σmax,σmin,k,start,finish)
    t1 = Float64[]
    t2 = Float64[]
    t3 = Float64[]
    t4 = Float64[]
    t5 = Float64[]
    t6 = Float64[]
    for n = start:finish
        push!(t1,10^(12)*σmax^k*tan(σmax)^k*cos(σmax)^n)
        push!(t2,10^12*σmin^k*tan(σmin)^k*cos(σmin)^n)
        push!(t3,10^12*σmax^k*tan(σmin)^k*cos(σmin)^n)
        push!(t4,10^12*σmin^k*tan(σmax)^k*cos(σmax)^n)
        push!(t5,10^12*2*σmax^(k/2)*σmin^(k/2)*tan(σmax)^(k/2)*tan(σmin)^(k/2)*cos(σmax)^(n/2)*cos(σmin)^(n/2))
        push!(t6,10^(12)*σmax^k*tan(σmin)^k*cos(σmin)^n+10^12*σmin^k*tan(σmax)^k*cos(σmax)^n)
    end
    plot(start:finish,[vec(t1),vec(t2),vec(t3),vec(t4),vec(t5),vec(t6)],label=["t1","t2","t3","t4","t5","t3+t4"])
end

function assumemeasure(η::Float64,σmax,σmin,k::Int)
    1/log(cos(σmin)/cos(σmax))*(-log(η/(1-η))/2+k*log(tan(σmax)/tan(σmin)))
end

function assume(σ1,σ2,σ3,σ4,k)
    t1 = Float64[]
    t2 = Float64[]
    t3 = Float64[]
    t4 = Float64[]
    t5 = Float64[]
    start = 200
    finish = 400
    for n = start:finish
        push!(t1,10^(30)*σ1^k*tan(σ1)^k*cos(σ1)^n)
        push!(t2,10^(30)*σ1^k*tan(σ2)^k*cos(σ2)^n)
        push!(t3,10^(30)*σ1^k*tan(σ4)^k*cos(σ4)^n)
    end
    plot(start:finish,[vec(t1),vec(t2),vec(t3)],label=["t1","t2","t3"])
end

function p1(k,ϵ,nbit)
    p1 = Float64[]
    for i in k:5000
        push!(p1,binomial(BigInt(i-1),BigInt(k-1))*sin(ϵ)^(2k)*cos(ϵ)^(2i-2k)/(2^nbit))
    end
    p1
end

function p2(σ,k,ϵ,nbit)
    p2 = Float64[]
    for i in k:5000
        push!(p2,binomial(BigInt(i-1),BigInt(k-1))*sin(ϵ*σ)^(2k)*cos(ϵ*σ)^(2i-2k)/(2^nbit))
    end
    p2
end

function plotsigma(eps,percent::Float64,nbit::Int,k::Int,ϵ)
    reg = uniform_state(nbit)
    t = Int[]
    p = Float64[]
    p11 = Float64[]
    p22 = Float64[]
    for i in eps
        Σ = Diagonal(append!([1,i],rand(Int(2^nbit)-2)./2))
        tmp = threshold(percent,Σ,reg,k,ϵ,2000)
        push!(t,tmp)
        pro = sum(exactprob(Σ,reg,k,ϵ)[1:tmp-k+1])
        push!(p,pro)
        p1t = sum(p1(k,ϵ,nbit)[1:tmp-k+1])
        push!(p11,p1t)
        p2t = sum(p2(i,k,ϵ,nbit)[1:tmp-k+1])
        push!(p22,p2t)
    end
    return [vec(p),vec(p11),vec(p22)]
end

function plotsigma(eps,Σ,percent::Float64,nbit::Int,k::Int,ϵ,τ)
    reg = uniform_state(nbit)
    t = Int[]
    p = Float64[]
    p11 = Float64[]
    p22 = Float64[]
    for i in m
        tΣ = Diagonal(diag(Σ).^(τ/i))
        tmp = threshold(percent,tΣ,reg,k,ϵ,2000)
        push!(t,tmp)
        pro = sum(exactprob(tΣ,reg,k,ϵ)[1:tmp-k+1])
        push!(p,pro)
        p1t = sum(p1(k,ϵ,nbit)[1:tmp-k+1])
        push!(p11,p1t)
        p2t = sum(p2(tΣ[2,2],k,ϵ,nbit)[1:tmp-k+1])
        push!(p22,p2t)
    end
    return [vec(p),vec(p11),vec(p22)]
end

function sortsigma(Σ,sort::Bool)
    ele = diag(Σ)
    if sort == true
        ele = sort!(ele,rev=true)./ele[1]
    else
        ele = ele./maximum(ele)
    end
    return Diagonal(ele)
end

function irotZZ(theta)
    Diagonal([exp(-theta/2),exp(theta/2),exp(theta/2),exp(-theta/2)])
end

ϵ = 0.5
Σ = sortsigma(irotZZ(5),true)
reg = uniform_state(2)
k = 1
t = threshold(0.95*0.95,Σ,reg,k,ϵ,2000)
assumemeasure(0.95*0.95,ϵ*1.,ϵ*0.7,k)
plotfidelity(Σ,reg,300,k,ϵ)
sum(exactprob(Σ,reg,k,ϵ)[1:t-k+1])
delta = 0.7:0.01:1.0
plot(delta,plotsigma(delta,0.95*0.95,2,k,ϵ))
assume(1,0.9,k,80,90)

function meanpro(nbit,k,ϵ)
    Σ = Diagonal(rand(2^nbit))
    ssum = Float64[]
    for i in 1:50
        reg = rand_state(nbit)
        x = statevec(reg)
        push!(ssum,sum(exactprob(Σ,reg,k,ϵ)[1:threshold(0.9,Σ,reg,k,ϵ,50000)-k+1]))
    end
    mean(ssum)
end

function testmeanpro(nbit)
    ssum = zeros(Float64,20,20)
    for k = 1:10, ϵ = 1:20
        ssum[k,ϵ] = meanpro(nbit,k,0.1+0.02*ϵ)
    end
    ssum
end

testmeanpro(3)
