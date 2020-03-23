using Flux
using Plots
using LaTeXStrings
using SpecialFunctions

py(y,θ,s) = 1/sqrt(s)*exp(-0.5*(y-θ)^2/s);
pθ(θ,α) = 1/sqrt(α)*exp(-0.5*θ^2/(α));
pα(α) = α^(-1); #Jeffrey


joint(y1,y2,θ,α,s) = py(y1,θ,s)*py(y2,θ,s)*pθ(θ,α)*pα(α);

plc=contour(9.5:0.01:13,8:0.01:400,(θ,α)->joint(11.,12.,θ,α,1.), xlabel=L"$\theta$", ylabel=L"$\alpha$", levels=6)

# elbo
#qθ = N(μ,σ) =>
#qα = iG(γ,δ) =>

elbo2(y,μ,σ,γ,δ) = -0.5.*((y[1]).^2 .+ (y[2]).^2) .+ μ.*(y[1] .+ y[2] .- μ./(2 .*σ)) .-
  (μ.^2 .- σ).*(1 .+ γ./(2 .*δ)-1 ./(2 .*σ)) .- μ.^2 ./(2 .*σ) .- 0.5.*log.(δ) .-
   (γ.-0.5).*digamma.(γ) .+ γ .+ log.(1 ./((σ.^(0.5).*gamma.(γ))));

μ = [1.0];
σ = [1.0];
γ = [1.0];
δ = [1.0];

using Distributions
loss_elbo(y) = -sum(elbo2(y,μ,σ,γ,δ))
mydata = Iterators.repeated(([11.0 ; 12.0] , ), 1000)
TH=zeros(4,10);
for epo=1:10
    global plc
    global TH
    Flux.train!(loss_elbo,Flux.params([μ,σ,γ,δ]),mydata,ADAM(0.01))
    plt(θ,α) = pdf(Normal(μ[1],sqrt(exp.(logσ[1]))),θ) .* pdf.(InverseGamma(γ[1],δ[1]), α)
    contour!(plc,10:0.01:14,0:0.1:1.5,plt,levels=5, linestyle=:dash)
    TH[:,epo].=vcat(μ,σ,γ,δ);
end
TH
gui()
