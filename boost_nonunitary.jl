using Yao
using LinearAlgebra
using Roots
using LaTeXStrings

function grover_step!(reg::AbstractRegister,oracle,U)
    apply!(reg |> oracle, reflect_circuit(U))
end

function reflect_circuit(gen::AbstractBlock{N}) where N
    reflect0 = control(N, -collect(1:N-1),N=>-Z)
    chain(gen',reflect0,gen)
end

function UΣ(Σ,ϵ)
    matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])
end

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

function teps(Σ,reg,ϵ)
    abs(expect(matblock(cos(ϵ*Σ)^2),reg))
end

function grover(t,Σ,Ui)
    nbit = Int(log(2,size(Σ,1))) + 1
    oracle = put(nbit,nbit=>Z)
    psi0 = zero_state(nbit-1) |> Ui
    real_state = join(zero_state(1) |> X, zero_state(nbit-1) |> Ui |> matblock(Σ) |> normalize!)
    ϵ = minimum(find_zeros(x->teps(Σ,psi0,x)-t,0,1))
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



t = 0.7500
#t = 0.9855
#t = 0.9891
nbit = 3
Ui = matblock(rand_unitary(1<<(nbit-1)))
Ui = matblock(rand_unitary(1<<(nbit-1)))
Σ = Diagonal(Complex.([1.,exp(-10),exp(-10),1.]))
zero_state(nbit-1) |> Ui |> statevec
plot(1:10,grover(t,Σ,Ui)[1],lw=2,xlabel="\$k\$",ylabel="\$p_0\$",label="\$t = $t\$")
plot!(1:10,grover(t,Σ,Ui)[2],lw=2,label="\$f\$")
savefig("D:\\Nonunitary\\proj.pdf")
