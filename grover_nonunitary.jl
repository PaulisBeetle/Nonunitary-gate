using Yao
using LinearAlgebra
using Roots
using LaTeXStrings
using Plots

function reflect_circuit(gen::AbstractBlock{N}) where N
    reflect0 = control(N,-collect(1:N-1),N=>-Z)
    chain(gen',reflect0,gen)
end

UΣ(Σ,ϵ) = matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])

function target_state(Σ::Array,ϵ)
    nbit = 2*size(Σ,1) + 1
    ngate = size(Σ,1)
    zero_state(nbit) |> put(nbit,(1,)=>H) |> repeat(nbit,H,2:2:nbit-1) |> repeat(nbit,X,3:2:nbit) |> put(nbit,(1,2)=>matblock(sin(ϵ*Σ[1]))) |> chain(nbit,[put((2i-2,2i)=>matblock(sin(ϵ*Σ[i]))) for i in 2:ngate]) |> normalize!
end

function deps(Σ,reg,ϵ)
    ngate = size(Σ,1)
    return abs(expect(chain(ngate+1,[put((i,i+1)=>matblock(sin(ϵ*Σ[i])^2)) for i in 1:ngate]),reg))
end

function rotzzdeps(W,τ)
    ngate = size(W,1)
    dd = []
    for eps = 0.1:0.1:12pi
        d = 1.
        for j = 1:ngate
            d = d*(sin(eps*exp(W[j]*τ))^2+sin(eps*exp(-W[j]*τ))^2)
        end
        d = d/2^(ngate)
        push!(dd,d)
    end
    dd
end

function sp(s,k)
    s = Complex(s)
    return abs((sqrt(s)+sqrt(s-1))^(2k+1)+(sqrt(s)-sqrt(s-1))^(2k+1))^2/4
end

Σ = [Diagonal(Complex.(rand(4))) for i in 1:5]
s = 1/4
nbit = 2*size(Σ,1) + 1
ngate = size(Σ,1)

plot([deps(Σ,uniform_state(ngate+1),ϵ) for ϵ in 0.01:0.01:2pi])
ϵ = minimum(find_zeros(x->deps(Σ,uniform_state(ngate+1),x)-d,0,2pi))
Ui = chain(put((1,)=>H),repeat(nbit,H,2:2:nbit-1))
oracle = control(nbit,3:2:nbit-2,nbit=>Z)
t = chain(nbit,[put((2i-2,2i,2i+1)=>UΣ(Σ[i],ϵ)) for i in 2:ngate])
gen = chain(Ui,put((1,2,3)=>UΣ(Σ[1],ϵ)),t)
reg = zero_state(nbit) |> gen
prob = []
reg |> control(nbit,3:2:nbit-2,nbit=>Z) |> reflect_circuit(gen)
measure(reg,(3,5,7),nshots=10)
t_state = target_state(Σ,ϵ)
p = abs(reg'*t_state)
push!(prob,p)

function findkofp1(s0::Float64,itersteps)
    error = zeros(itersteps)
    kstar = zeros(itersteps)
    for i = 1:itersteps
        for k = 1:100000
            global deltanow = 1/(i^4) - sp(s0, k)
            global deltanext = 1/(i^4) - sp(s0,k+1)
            if abs(deltanow) < abs(deltanext)
                kstar[i] = k
                error[i] = deltanow
                break
            end
        end
        s0 = s0/2
    end
    return error,kstar
end


s0 = 0.5
findkofp1(s0,50)
