using Yao
using LinearAlgebra
using Roots
using LaTeXStrings
using Plots

function grover_step!(reg::AbstractRegister,oracle,U)
    apply!(reg |> oracle, reflect_circuit(U))
end

function reflect_circuit(gen::AbstractBlock{N}) where N
    reflect0 = control(N, -collect(1:N-1),N=>-Z)
    chain(gen',reflect0,gen)
end


ΣW(i,W,τ) = W[i] > 0 ?  Diagonal(Complex.([1.,exp(-2*W[i]*τ),exp(-2*W[i]*τ),1.])) : Diagonal(Complex.([exp(2*W[i]*τ),1.,1.,exp(2*W[i]*τ)]))
UΣ(Σ,ϵ) = matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])

function target_state(Σ,ϵ,Ui)
    nbit = Int(log(2,size(Σ,1)))
    reg =  zero_state(nbit) |> Ui |> matblock(sin(ϵ*Σ)) |> normalize!
    join(zero_state(1) |> X,reg)
end

function probfail(Σ,reg,ϵ,k)
    x = statevec(reg)
    t = x'*cos(ϵ*Σ)^2*x
    t1 = -sqrt(Complex.(t*(t-1)))*(2t-2*sqrt(Complex.(t*(t-1)))-1)^k
    t2 = sqrt(Complex.(t*(t-1)))*(2t+2*sqrt(Complex.(t*(t-1)))-1)^k
    t3 = t*(2t-2*sqrt(Complex.(t*(t-1)))-1)^k
    t4 = t*(2t+2*sqrt(Complex.(t*(t-1)))-1)^k
    return abs((t1+t2+t3+t4)^2/(4t))
end

function probfail(t,k)
    t1 = -sqrt(Complex.(t*(t-1)))*(2t-2*sqrt(Complex.(t*(t-1)))-1)^k
    t2 = sqrt(Complex.(t*(t-1)))*(2t+2*sqrt(Complex.(t*(t-1)))-1)^k
    t3 = t*(2t-2*sqrt(Complex.(t*(t-1)))-1)^k
    t4 = t*(2t+2*sqrt(Complex.(t*(t-1)))-1)^k
    return abs((t1+t2+t3+t4)^2/(4t))
end

function teps(Σ,reg,ϵ)
    abs(expect(matblock(cos(ϵ*Σ)^2),reg))
end

# generates all reflect_circuit in circuit and store them in rarray
function gen_reflect_circuit(W,τ,ϵ::Array)
    n = size(W,1)
    rarray = []
    for i = 1:n
        nbit = 2i + 1
        r = chain(nbit)
        push!(r,put(nbit,(i,2i,2i+1)=>(UΣ(ΣW(i,W,τ),ϵ[i]))'))
        push!(r,put((2i,)=>H))
        push!(r,control(nbit,(-(2i),-(2i+1)),(1:nbit-2)=> i == 1 ? -X : rarray[i-1]))
        push!(r,put((2i,)=>H))
        push!(r,put((i,2i,2i+1)=>UΣ(ΣW( i,W,τ),ϵ[i])))
        push!(rarray,r)
    end
    rarray
end

function interact(i,W,τ,ref,ϵ)
    nbit = 2i + 1
    c = chain(nbit)
    push!(c,put(nbit,(i,2i,2i+1)=>UΣ(ΣW(i,W,τ),ϵ[i])))
    push!(c,put(nbit,(nbit,)=>Z))
    push!(c,put(nbit,(1:nbit)=>ref))
    c
end


#single layer
function grover(t,Σ,Ui)
    nbit = Int(log(2,size(Σ,1))) + 1
    oracle = put(nbit,nbit=>Z)
    psi0 = zero_state(nbit-1) |> Ui
    real_state = join(zero_state(1) |> X, zero_state(nbit-1) |> Ui |> matblock(Σ) |> normalize!)
    ϵ = minimum(find_zeros(x->teps(Σ,psi0,x)-t,0,1))
    ϵ = π/4
    #k = 1:10
    #plot(k,[probfail(Σ,psi0,ϵ,i) for i in k],lw=2)
    oracle = put(nbit,(nbit,)=>Z)
    gen = chain(nbit,put(nbit,(1:nbit-1)=>Ui),put(nbit,(1:nbit)=>UΣ(Σ,ϵ)))
    reg = zero_state(nbit) |> gen
    t_state = target_state(Σ,ϵ,Ui)
    probf = []
    fidel = []

    for i = 1:10
        grover_step!(reg,oracle,gen)
        overlap = abs(reg'*t_state)
        fid = abs(reg'*real_state)
        push!(probf,1-abs2(overlap))
        push!(fidel,fid)
        println("step $(i), overlap = $overlap, prob = $(abs2(overlap)), fid = $fid")
    end
    probf,fidel
end

function zeropeta(k)
    return cos(π/(4k+2))^2
end

# single interact test code
t = 0.7500
#t = 0.9855
#t = 0.9891
nbit = 3
Ui = repeat(nbit-1,H,(1:nbit-1))
Σ = Diagonal(Complex.([exp(10),exp(-10),exp(-10),exp(10)]))
zero_state(nbit-1) |> Ui |> statevec
plot(1:10,grover(t,Σ,Ui)[1],lw=2,xlabel="\$k\$",ylabel="\$p_0\$",label="\$t = $t\$")
plot!(1:10,grover(t,Σ,Ui)[2],lw=2,label="\$f\$")
savefig("D:\\Nonunitary\\proj.pdf")

#multi-interact
W = [1.,-1.]
ϵ = fill(π/4,2)
τ = 10
ref = gen_reflect_circuit(W,τ,ϵ)

ntot = 5
circuit = chain(ntot)
push!(circuit,put((1,2,3)=>UΣ(ΣW(1,W,τ),ϵ[1])))
push!(circuit,put((3,)=>Z))
push!(circuit,put((1,2,3)=>ref[1]))
push!(circuit,put((2,4,5)=>UΣ(ΣW(2,W,τ),ϵ[2])))
push!(circuit,put((5,)=>Z))
push!(circuit,put((2,4,5)=>(UΣ(ΣW(2,W,τ),ϵ[2]))'))
push!(circuit,put((4,)=>H))
G1 = chain(3,put((3,)=>Z),put((1,2,3)=>ref[1]))
GRG = chain(3,put((1,2,3)=>G1'),put((1,2,3)=>ref[1]),put((1,2,3)=>G1))
#GRG = chain(3,put((1,2,3)=>ref[1]))
push!(circuit,control((-4,-5),(1,2,3)=>GRG))
push!(circuit,put((4,)=>H))
push!(circuit,put((2,4,5)=>UΣ(ΣW(2,W,τ),ϵ[2])))

reg = zero_state(5)
reg |> repeat(5,H,(1,2,4))
reg |> circuit
measure(reg,5,nshots=50)

ΣW(2,W,τ)
ΣW(1,W,τ)
