using Flux
using Plots
using LaTeXStrings
using SpecialFunctions

py1(y1,θ) = exp(-0.5*(y1-θ)^2);
py2(y2,θ) = exp(-0.5*(y2-θ)^2);
pθ(θ,α) = 1/sqrt(α)*exp(-0.5*θ^2/(α));
pα(α) = α^(-1); #Jeffrey

joint(y1,y2,θ,α) = py1(y1,θ)*py2(y2,θ)*pθ(θ,α)*pα(α);

plc=contour(9.5:0.1:13,8:1:400,(θ,α)->joint(11.,12.,θ,α), xlabel=L"$\theta$", ylabel=L"$\alpha$", levels=6)

# elbo
#qθ = N(μ,σ) => σ is variance!
#qα = iG(γ,δ) =>

ElogiG(γ,δ)= log.(δ) - digamma.(γ)

using Distributions
nEntq_θ(σ) = -0.5*log.(σ)
nEntq_α(γ,δ) = -γ.-log.(δ.*gamma.(γ)).+(1 .+ γ).*digamma.(γ)
Elogp_y1(y1,μ,σ) = -0.5*(y1.^2 .-2*y1.*μ.+μ.^2 .+σ)
Elogp_y2(y2,μ,σ) = -0.5*(y2^.2 .-2*y2.*μ.+μ.^2 .+σ)
Elogp_θ(μ,σ,γ,δ) = -0.5*(μ.^2 .+ σ).*γ./δ - 0.5*ElogiG(γ,δ)
Elogp_α(γ,δ) = -ElogiG(γ,δ)



elbo2(y1,y2,μ,σ,γ,δ) = sum(Elogp_y1(y1,μ,σ) + Elogp_y2(y2,μ,σ) +
<<<<<<< HEAD
 Elogp_θ(μ,σ,γ,δ) + Elogp_α(γ,δ) - Entq_θ(σ) - Entq_α(γ,δ))
=======
 Elogp_θ(μ,σ,γ,δ) + Elogp_α(γ,δ) - nEntq_θ(σ) - nEntq_α(γ,δ))
>>>>>>> 88321c08be2ed1aa15cdd6db7add2f9f60d35279

μ = [0.0001];
sσ = [.01]; # sqrtae root of σ
sγ = [1.5]; # square root of γ
sδ = [4.0];

using Distributions
mydata = Iterators.repeated(([11.;12.],), 100)
TH=zeros(4,13);
for epo=1:13
    global plc
    global TH
    Flux.train!((x)->-elbo2(x[1],x[2], μ, sσ.^2, sγ.^2, sδ.^2),Flux.params([μ,sσ,sγ,sδ]),mydata,ADAM(0.01))
    TH[:,epo].=vcat(μ,sσ.^2,sγ.^2,sδ.^2);
end
TH

plt(θ,α) = pdf(Normal(μ[1],sσ[1]),θ) .* pdf.(InverseGamma(sγ[1].^2,sδ[1].^2), α)
contour!(plc,9.5:0.1:13,8:1:400,(θ,α)->plt(θ,α), xlabel=L"$\theta$", ylabel=L"$\alpha$", levels=6, linestyle=:dash)
# contour(10:0.01:14,0:0.1:1.5,plt,levels=5, linestyle=:dash)
# gui()
