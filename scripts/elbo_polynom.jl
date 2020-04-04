using Flux
using Plots
using LaTeXStrings
using SpecialFunctions
using LinearAlgebra
using Distributions

# define LS

x = 1:2;
y = x.^2 .+ 1 .+ 1.0*randn(size(x));
X=[ones(size(x)) x.^2 x.^3 x.^4 x.^5];

#initial values of optimization
θh = randn(5,1);
A = randn((5,5));
γ = [2.0];
δ = [1.0];

# define loss function
Entq(Σ) = -0.5*log(det(Σ))
<<<<<<< HEAD
Entq(γ,δ) = -γ.-log.(δ.*gamma.(γ)).+(1 .-γ).*digamma.(γ)
Elogp(X,y,θh,Σ,γ,δ) = 0.5*(y'*y .- θh'*X'*y .- y'*X*θh .+
tr((X'*X + (γ[1]./ δ[1])*I)*(θh*θh'+Σ)) .- 7*(digamma.(γ) .- log.(δ)))
=======
Entq(γ,δ) = -γ.-log.(δ.*gamma.(γ)).+(1 .+γ).*digamma.(γ)
Elogp(X,y,θh,Σ,γ,δ) = 0.5*(y'*y .- θh'*X'*y .- y'*X*θh .+
tr((X'*X+δ[1]./(γ[1].-1)*I)*(θh*θh'+Σ)) .+ log.(δ) .- digamma.(γ))
>>>>>>> 091b9bcc1e6b90cf718940f561fdb3b8a01ab123
loss(X,y,θh,Σ,γ,δ)= Entq(Σ)+sum(Entq(γ,δ))+sum(Elogp(X,y,θh,Σ,γ,δ))


data = Iterators.repeated((X,y),1000)
Flux.train!((X,y)->loss(X,y,θh,A*A',γ,δ),[θh,A,γ,δ],data,ADAM(0.01))

scatter(x,y)
z=0:0.01:3
zv = θh[1] .+ θh[2].*z.^2 .+ θh[3].*z.^3 .+ θh[4].*z.^4 .+ θh[5].*z.^5;
plot!(z, zv)

θh
Σ = A'*A
γ
δ