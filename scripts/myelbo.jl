using Flux
using Plots
using LaTeXStrings
using SpecialFunctions

py1(y1,θ,s) = 1/sqrt(s)*exp(-0.5*(y1-θ)^2/s);
py2(y2,θ,s) = 1/sqrt(s)*exp(-0.5*(y2-θ)^2/s);
pθ(θ,α) = 1/sqrt(α)*exp(-0.5*θ^2/(α));
pα(α) = α^(-1); #Jeffrey


joint(y1,y2,θ,α,s) = py1(y1,θ,s)*py2(y2,θ,s)*pθ(θ,α)*pα(α);

plc=contour(9.5:0.01:13,8:0.01:400,(θ,α)->joint(11.,12.,θ,α,1.), xlabel=L"$\theta$", ylabel=L"$\alpha$", levels=6)

# elbo
#qθ = N(μ,σ) =>
#qα = iG(γ,δ) =>

using Distributions
Entq_θ(σ) = -0.5*log.(σ)
Entq_α(γ,δ) = -γ.-log.(δ.*gamma.(γ)).+(1 .+ γ).*digamma.(γ)
Elogp_y1(y1,μ,σ) = -0.5*(y1.^2 .-2*y1.*μ.+μ.^2 .+σ)
Elogp_y2(y2,μ,σ) = -0.5*(y2^.2 .-2*y2.*μ.+μ.^2 .+σ)
Elogp_θ(μ,σ,γ,δ) = -0.5*(μ.^2 .+ σ).*γ./δ
Elogp_α(γ,δ) = digamma.(γ).-log.(δ)



elbo2(y1,y2,μ,σ,γ,δ) = sum(Elogp_y1(y1,μ,σ) + Elogp_y2(y2,μ,σ) +
 Elogp_θ(μ,σ,γ,δ) + Elogp_α(γ,δ) - Entq_θ(σ) - Entq_α(γ,δ))

μ = [10.0];
σ = [100.0];
γ = [1.5];
δ = [4.0];

using Distributions
loss_elbo(y1,y2) = -elbo2(y1,y2,μ,σ,γ,δ)
mydata = Iterators.repeated(([11.0;2.0],), 1000)
TH=zeros(4,13);
for epo=1:13
    global plc
    global TH
    Flux.train!((x)->loss_elbo(x[1],x[2]),Flux.params([μ,σ,γ,δ]),mydata,ADAM(0.001))
    plt(θ,α) = pdf(Normal(μ[1],sqrt(σ[1])),θ) .* pdf.(InverseGamma(γ[1],δ[1]), α)
    contour!(plc,10:0.01:14,0:0.1:1.5,plt,levels=5, linestyle=:dash)
    TH[:,epo].=vcat(μ,σ,γ,δ);
end
TH
gui()
