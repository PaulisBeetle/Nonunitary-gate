using Yao
using Statistics
@const_gate ZZ = mat(kron(Z,Z))

function turplesort(k::Int,l::Int,n::Int)
    s = zeros(Int,n,n,1)
    st = 1
    for i = 1:n, j = i+1:n
        s[i,j] = st
        st +=1
    end
    s[k,l]
end

function check(k::Int,W::AbstractMatrix)
    n = size(W,1)
    flag = 0
    for i = 1:n, j = i+1:n
        if turplesort(i,j,n) == k
            if W[i,j] != 0
                flag = 1
                break
            end
        end
    end
    return flag
end

function check(W::AbstractMatrix)
    n = size(W,1)
    cnt = 0
    for i = 1:n, j = i+1:n
        if W[i,j] != 0
            cnt += 1
        end
    end
    cnt
end

function nonmaxcut_circuit(W::AbstractMatrix,τ,ϵ)
    nphy = size(W,1)
    nanc = Int(nphy*(nphy-1)/2)
    nbit = nphy + nanc
    ab = chain(nbit)
    for i = 1:nphy, j=i+1:nphy
        if W[i,j]!=0
            Σ = mat(rot(ZZ,-im*W[i,j]*τ))
            Σ = matblock(Σ)
            #push!(ab,put(nbit,(turplesort(i,j,nphy)+nphy,i,j)=>TimeEvolution(kron(Y,Σ),ϵ)))
            push!(ab,put(nbit,(i,j,turplesort(i,j,nphy)+nphy)=>matblock(exp(-im*ϵ*mat(kron(Y,Σ))))))
        end
    end
    for i = nphy+1:nbit
        push!(ab,Measure(nbit,locs=(i,),collapseto=bit"0"))
    end
    return ab
end

function maxcut_h(W::AbstractMatrix)
    nbit = size(W,1)
    ab = Add{nbit}()
    for i = 1:nbit, j = i+1:nbit
        if W[i,j]!=0
            push!(ab,put(nbit,(i,j)=>0.5*W[i,j]*ZZ))
        end
    end
    return ab
end

function act(nphy,W,τ,ϵ)
    nanc = Int(nphy*(nphy-1)/2)
    nbit = nphy+nanc
    wmax = maximum(maximum(W,dims=1))
    Wn = W./wmax
    c = nonmaxcut_circuit(Wn,τ,ϵ)
    h = maxcut_h(W)
    reg_phy = uniform_state(nphy)
    reg_anc = zero_state(nanc)
    reg     = join(reg_anc,reg_phy)
    Nt = 10/τ*2000
    flag = zeros(Int64,nanc)
    mblocks = collect_blocks(Measure,c)
    for i in 1:Nt
        reg |> c
        @show probs(reg_phy)
        for j in 1:nanc
            if check(j,W)!= 0
                if flag[j] == Int(10/τ)
                    flag[j] = -1
                else
                    if (mblocks[j].results)[1] == 1
                        flag[j] += 1
                    end
                end
            end
        end
        if sum(flag) == -1*check(W)
            break
        end
    end
    reg
end

nphy = 5
W = [0 5 2 1 0;
     5 0 3 2 0;
     2 3 0 0 0;
     1 2 0 0 4;
     0 0 0 4 0]
τ = 1/4
act(nphy,W,τ,0.5)
op_tl = reg_phy |> probs
config = argmax(op_tl) - 1
