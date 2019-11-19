using Yao
using Plots
using LinearAlgebra
using Random
using Statistics
using HypergeometricFunctions

rng = Random.GLOBAL_RNG
II = [1. 0;0 1.]
@const_gate ZZ = mat(kron(Z,Z))

function testfidelity(Σ,regn,k::Int,ϵ)
    nbit = Int(log(2,size(statevec(regn))[1]))+1
    Nt = 5000*k/(ϵ^2)
    Σ = Complex.(Σ)
    aΣ = [cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)]
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
    f0 = (Yao.fidelity(reg,join(zero_state(1),regt)))[1]
    if num == k
        return measuretimes,f0
    else
        return measuretimes, false
    end
end

function fidmes(Σ,reg,k::Int,ϵ::Float64,minn,maxn)
    x = statevec(reg)
    x = BigFloat.(x)
    diff = (x'*BigFloat.(Σ)^(2k+4)*x)*(x'*BigFloat.(Σ)^(2k)*x)-(x'*BigFloat.(Σ)^(2k+2)*x)^2
    A = (x'*BigFloat.(Σ)^(2k)*x)^2
    #approx = [sqrt(abs(1-(n/2-k/3)^2*diff*ϵ^4/A)) for n in k:2000]
    rstate = BigFloat.(Σ)^k*x./sqrt((x'*BigFloat.(Σ)^(2k)*x))
    exact = [abs(rstate'*sin(ϵ*BigFloat.(Σ))^(k)*cos(ϵ*BigFloat.(Σ))^(n-k)*x/(sqrt(x'*sin(ϵ*BigFloat.(Σ))^(2k)*cos(ϵ*BigFloat.(Σ))^(2n-2k)*x)))  for n in minn:maxn]
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
    histogram(f,bins=0.0:0.05:1.1)
    title!("Distribution of fidelity after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//fidelityhis_$(numk)_eps$ϵ")
    histogram(m,bins=0:100:12000,title="Distribution of measure times after $(length(f)) samples, eps=$ϵ")
    savefig("D://Nonunitary//measuretimeshis_$(numk)_eps$ϵ")
    scatter(results,title="fidelity-measuretimes, k=$numk, eps=$ϵ",xlabel="measuretimes",ylabel="fidelity")
    plot!(numk:12000,fidmes(Σ,reg2,numk,ϵ,numk,12000))
    savefig("D://Nonunitary//fidmes_k=$(numk)_eps=$(ϵ)")
    f,m
    #=histogram(m,bins=numk:50:)=#
end


function exactprob(Σ::Diagonal,reg,k::Int,ϵ::Float64,minn,n::Int)
    x = statevec(reg)
    p = Float64[]
    for i in minn:n
        push!(p,abs((binomial(BigInt(i-1),BigInt(k-1))*x'*sin(BigFloat.(ϵ*Σ))^(2*k)*cos(BigFloat.(ϵ*Σ))^(2*i-2*k)*x)))
    end
    p
end

function measuretimes(Σ::Diagonal,reg,k::Int,ϵ)
    x = statevec(reg)
    return abs(k*x'*(sin(ϵ*Σ)^(-2))*x)
end

function exactfid(Σ::Diagonal,reg,n::Int,k::Int,ϵ)
    x = statevec(reg)
    num = (x'*BigFloat.(Σ)^k*sin(ϵ*BigFloat.(Σ))^k*cos(ϵ*BigFloat.(Σ))^(n-k)*x)^2
    den = (x'*BigFloat.(Σ)^(2k)*x)*(x'*sin(ϵ*BigFloat.(Σ))^(2k)*cos(ϵ*BigFloat.(Σ))^(2n-2k)*x)
    return sqrt(abs(num/den))
end

function threshold(percent::Float64,Σ,reg,k::Int,ϵ,minn,maxn)
    function f(x)
        return exactfid(Σ,reg,x,k,ϵ)-percent
    end
    maxifid = exactfid(Σ,reg,k,k,ϵ)
    for i in minn:maxn
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


function meanpro(nbit,Σ,k,ϵ)
    ssum = Float64[]
    for i in 1:100
        reg = rand_state(nbit)
        x = statevec(reg)
        push!(ssum,sum(exactprob(Σ,reg,k,ϵ,threshold(0.9,Σ,reg,k,ϵ,100000))))
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

function exactprobconstn(Σ::Diagonal,reg,n::Int,ϵ::Float64)
    x = statevec(reg)
    p = Float64[]
    for i in 1:n
        push!(p,abs(binomial(BigInt(n-1),BigInt(i-1))*x'*sin(ϵ*Σ)^(2i)*cos(ϵ*Σ)^(2n-2i)*x))
    end
    p
end

function contrast(Σ,reg,k,ϵ,maxn)
    p = exactprob(Σ,reg,k,ϵ,maxn)
    f = fidmes(Σ,reg,k,ϵ,maxn)
    p = p./maximum(p)
    plot(k:maxn,[vec(p),vec(f)],label=["prob","fid-mes"])
    savefig("D://Nonunitary//contrast_$(k)_$(ϵ)")
    return
end

function nandk(Σ,reg,ϵ,maxk)
    na = Int[]
    for k =1:maxk
        push!(na,threshold(0.95*0.95,Σ,reg,k,ϵ,1000000))
    end
    na
end

function coeff(σ,ϵ,η)
    return log(tan(ϵ)/tan(ϵ*σ))/log(cos(ϵ*σ)/cos(ϵ)),-1/(2*log(cos(ϵ*σ)/cos(ϵ)))*log(η/(1-η))
end

function decompose(Σ,reg,m,k,ϵ)
    nm = m^3
    Σ = Σ./maximum(Σ)
    mΣ = Σ.^(1/nm)
    plotfidelity(mΣ,reg,400,nm*k,ϵ)
end

function changeps(Σ,k)
    p = Float64[]
    eps = 0.001:0.001:0.01
    for i in eps
        t = threshold(0.95*0.95,Σ,uniform_state(2),k,i,10000000)
        push!(p,sum(exactprob(Σ,uniform_state(2),k,i,50000000)[k:t-k+1]))
    end
    p
end

function embedproj(nphy,ϵ)
    Nsample = 200
    Nmeasure = 4000
    Σ = Diagonal(Complex.([1,0]))
    u = matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])
    m = Measure(nphy+1,locs=(nphy+1,),collapseto=bit"0")
    marray = Int[]
    for i = 1:Nsample
        reg = join(zero_state(1),rand_state(nphy))
        c = chain(nphy+1,put(nphy+1,(nphy,nphy+1)=>u),m)
        for j = 1:Nmeasure
            reg |> c
            if m.results[1] == 1
                push!(marray,j)
                break
            end
        end
    end
    marray
end

function estimatemn(Σ,m,ϵ)
    σ = minimum(diag(Σ))
    n = (m-1)/(1-cos(ϵ*σ^(1/m))^2)
end


function decompexactfid(Σ,reg,m,n,ϵ)
    x = statevec(reg)
    num = (x'*Σ*sin(BigFloat.(ϵ*Σ.^(1/m)))^m*cos(BigFloat.(ϵ*Σ.^(1/m)))^(n-m)*x)^2
    dem = (x'*Σ^2*x)*(x'*sin(BigFloat.(ϵ*Σ.^(1/m)))^(2m)*cos(BigFloat.(ϵ*Σ.^(1/m)))^(2n-2m)*x)
    return sqrt(abs(num/dem))
end

function decompmaxn(Σ,m,ϵ)
    σ = minimum(diag(Σ))
    return round((m+1)/(1-cos(ϵ*BigFloat(σ^(1/m)))^2))
end

function decomplot(Σ,reg,m,ϵ,minn,maxn)
    prob = exactprob(Σ.^(1/m),reg,m,ϵ,minn,maxn)
    plot(minn:maxn,prob./maximum(prob))
    plot!(minn:maxn,fidmes(Σ.^(1/m),reg,m,ϵ,minn,maxn))
    savefig("D://Nonunitary//decomplot_$(m)_$ϵ")
end



part = Float64[]
mbar = Float64[]
for nbit = 5:1:25
     m = embedproj(nbit,0.4)
    push!(part,size(m,1)/200)
    push!(mbar,mean(m))
end

Σ = Diagonal([exp(-1/4),1,1,exp(-1/4)])
m = 100
ϵ = π/2
decomplot(Σ.^(1/m),uniform_state(2),m,ϵ,m,2000)
plot(250:1000,exactprob(Σ.^(1/m),uniform_state(2),m,ϵ,250,1000)./maximum(exactprob(Σ.^(1/m),uniform_state(2),m,ϵ,250,500)))
plot!(m:2000,fidmes(Σ.^(1/m),uniform_state(2),m,ϵ,m,2000))
estimatemn(Σ.^(1/m),m,ϵ)
exactfid(Σ.^(1/m),uniform_state(2),m,m,ϵ)
plotfidelity(Σ.^(1/m),uniform_state(2),200,2*m,ϵ)
