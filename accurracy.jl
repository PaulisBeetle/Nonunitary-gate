using Yao

function checkmsin(x,m,ϵ)
    a = BigFloat(sin(ϵ*BigFloat(x)^(1/m)))^m
    b = BigFloat(BigFloat(ϵ)^m)*x
    a/b
end

function decomplot(Σ,reg,m,ϵ,minn,maxn)
    prob = exactprob(Σ.^(1/m),uniform_state(2),m,ϵ,minn,maxn)
    plot(minn:maxn,prob./maximum(prob))
    plot!(minn:maxn,fidmes(Σ.^(1/m),uniform_state(2),m,ϵ,minn,maxn))
    savefig("D://Nonunitary//decomplot_$(m)_$ϵ")
end

function checkksin(x,k,ϵ)
    a = sin(ϵ*x)^k
    b = ϵ^k*x^k
    return a/b
end

function checkmnum(Σ,m,n,ϵ)
    x = statevec(uniform_state(2))
    num = (x'*Σ*sin(ϵ*BigFloat.(Σ)^(1/m))^(m)*cos(ϵ*BigFloat.(Σ)^(1/m))^(n-m)*x)^2
    den = x'*Σ^2*x*x'*sin(ϵ*BigFloat.(Σ)^(1/m))^(2m)*cos(ϵ*BigFloat.(Σ)^(1/m))^(2n-2m)*x
    return sqrt(num/den)
end

function testassume1(Σ,m)
    x = statevec(uniform_state(2))
    num = x'*Σ^(2+4/m)*x*x'Σ^2*x-(x'*Σ^(2+2/m)*x)^2
    den = (x'*Σ^2*x)^2
    return abs(m^2*num/den)
end

function testassum2(Σ,m)
    x = statevec(uniform_state(2))
    num = (x'*Σ*sin(π/2*BigFloat.(Σ)^(1/m))^(2m)*x)^2
    den = (x'*Σ^2*x)*(x'*sin(π/2*Σ^(1/m))^(4m)*x)
    return sqrt(abs(num/den))
end

function testassum3(Σ)
    x = statevec(uniform_state(2))
    return sqrt(abs(x'*Σ*x)^2/abs(x'*Σ^2*x))
end

N = 10
t = 10
ϵ = π/2
Σ = Diagonal([exp(-1/(N^2*t^2)),1,1,exp(-1/(N^2*t^2))])
plotfidelity(Σ,uniform_state(2),200,100,ϵ)
Σ = Diagonal(rand(4))
testassum3(Σ)
