using Yao
using LinearAlgebra
using Roots
using LaTeXStrings
using Plots

function reflect_circuit(gen::AbstractBlock{N}) where N
    reflect0 = control(N,-collect(1:N-1),N=>-Z)
    chain(gen',reflect0,gen)
end

ΣW(i,W,τ) = W[i] > 0 ?  Diagonal(Complex.([1.,exp(-2*W[i]*τ),exp(-2*W[i]*τ),1.])) : Diagonal(Complex.([exp(2*W[i]*τ),1.,1.,exp(2*W[i]*τ)]))
UΣ(Σ,ϵ) = matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])

function target_state(Σ::Array,ϵ)
    nbit = 2*size(Σ,1) + 1
    ngate = size(Σ,1)
    zero_state(nbit) |> put(nbit,(1,)=>H) |> repeat(nbit,H,2:2:nbit-1) |> repeat(nbit,X,3:2:nbit) |> put(nbit,(1,2)=>matblock(sin(ϵ*Σ[1]))) |> chain(nbit,[put((2i-2,2i)=>matblock(sin(ϵ*Σ[i]))) for i in 2:ngate]) |> normalize!
end

function real_state(Σ::Array)
    nbit = 2*size(Σ,1) + 1
    ngate = size(Σ,1)
    zero_state(nbit) |> put(nbit,(1,)=>H) |> repeat(nbit,H,2:2:nbit-1) |> repeat(nbit,X,3:2:nbit) |>  put(nbit,(1,2)=>matblock(Σ[1])) |> chain(nbit,[put((2i-2,2i)=>matblock(Σ[i])) for i in 2:ngate]) |> normalize!
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

function randcouple(N)   #random coupling -1 and 1
    [rand(1)[1] < 0.5 ? -1 : 1 for i = 1:N]
end


function grovergsimag(N::Int,τ::Float64)
    nbit = 2*N+1
    W = randcouple(N)
    Σ = [ΣW(i,W,τ) for i in 1:N]
    ϵ = π/2
    s = deps(Σ,uniform_state(N+1),ϵ)
    k = (π/asin(sqrt(s)) - 2)/4                                                 #The number of Grover iterations
    Ui = chain(put((1,)=>H),repeat(nbit,H,2:2:nbit-1))                          #physical qubit 1,2,4,6,8,...
    oracle = control(nbit,3:2:nbit-2,nbit=>Z)                                   #ancillary qubit 3,5,7,9,...
    t = chain(nbit,[put((2i-2,2i,2i+1)=>UΣ(Σ[i],ϵ)) for i in 2:N])
    gen = chain(Ui,put((1,2,3)=>UΣ(Σ[1],ϵ)),t)
    reg = zero_state(nbit) |> gen                                               #generate circuit
    prob = []                                                                   #Success probability
    fid = []                                                                    #Fidelity between output_state and real_state
    t_state = target_state(Σ,ϵ)
    r_state = real_state(Σ)                                                     # target_state is nearly the same as real_state
    ######Grover iteration######
    for i = 1:Int(round(k))
        reg |> oracle |> reflect_circuit(gen)
        push!(prob,abs(reg'*t_state))
        push!(fid,fidelity(r_state,reg)[1])
    end
    ###########################
    prob,fid,reg
end


N = 10   #lattice size
τ = 10.   #imaginary time :: FLoat64
prob,fid,reg = grovergsimag(N,τ)
plot(prob)
reg_phy = measure_remove!(reg,3:2:2*N+1) #remove ancillary qubit
