using Yao
using LinearAlgebra
using Statistics

function sample(nphy,Σ,ϵ,nsample)
    Nt = 20000
    m = Measure(nphy+1,locs=(nphy+1,),collapseto=bit"0")
    Σ = Complex.(Σ)
    mΣ = matblock([cos(ϵ*Σ) -sin(ϵ*Σ); sin(ϵ*Σ) cos(ϵ*Σ)])
    narray = zeros(nsample)
    for i = 1:nsample
        reg_phy = rand_state(nphy)
        n = 0
        reg = join(zero_state(1),reg_phy)
        c = chain(nphy+1,put(nphy+1,(nphy,nphy+1)=>mΣ),m)
        for j = 1:Nt
            reg |> c
            n = n+1
            if m.results[1] == 1 && j < Nt
                push!(narray,n)
                break
            end
        end
    end
    narray
end

mtimes = Float64[]
for nphy = 1:15
    push!(mtimes,mean(sample(nphy,Diagonal([1,0]),0.4,50)))
end
