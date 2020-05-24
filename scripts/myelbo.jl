using Flux
using Plots
using LaTeXStrings
using SpecialFunctions

py1(y1,θ) = exp(-0.5*(y1-θ)^2);
py2(y2,θ) = exp(-0.5*(y2-θ)^2);
pθ(θ,α) = 1/sqrt(α)*exp(-0.5*(θ^2)/α);
pα(α) = α^(-1); #Jeffrey


joint(y1,y2,θ,α) = py1(y1,θ)*py2(y2,θ)*pθ(θ,α)*pα(α);

plc=contour(1.3:0.01:4.5,0.0:0.01:28.0,(θ,α)->joint(3.0,4.0,θ,α), xlabel=L"$\theta$", ylabel=L"$\alpha$", levels=5)

# elbo
#qθ = N(μ,σ) =>
#qα = iG(γ,δ) =>

using Distributions
Entq_θ(σ) = -0.5*log.(σ)
Entq_α(γ,δ) = -γ.-log.(δ.*gamma.(γ)).+(1 .+ γ).*digamma.(γ)
Elogp_y1(y1,μ,σ) = -0.5*(y1.^2 .-2*y1.*μ.+μ.^2 .+σ)
<<<<<<< HEAD
Elogp_y2(y2,μ,σ) = -0.5*(y2.^2 .-2*y2.*μ.+μ.^2 .+σ)
Elogp_θ(μ,σ,γ,δ) = -0.5*((μ.^2 .+ σ).*(γ./δ) .+ log.(δ) .- digamma.(γ))
=======
Elogp_y2(y2,μ,σ) = -0.5*(y2^.2 .-2*y2.*μ.+μ.^2 .+σ)
Elogp_θ(μ,σ,γ,δ) = -0.5*(μ.^2 .+ σ).*γ./δ
>>>>>>> 091b9bcc1e6b90cf718940f561fdb3b8a01ab123
Elogp_α(γ,δ) = digamma.(γ).-log.(δ)



elbo2(y1,y2,μ,σ,γ,δ) = sum(Elogp_y1(y1,μ,σ) + Elogp_y2(y2,μ,σ) +
 Elogp_θ(μ,σ,γ,δ) + Elogp_α(γ,δ) - Entq_θ(σ) - Entq_α(γ,δ))

μ = [10.0];
<<<<<<< HEAD
σ = [10.0];
=======
σ = [100.0];
>>>>>>> 091b9bcc1e6b90cf718940f561fdb3b8a01ab123
γ = [1.5];
δ = [4.0];

using Distributions
loss_elbo(y1,y2) = -elbo2(y1,y2,μ,σ,γ,δ)
<<<<<<< HEAD
mydata = Iterators.repeated(([3.0;4.0],), 1000)
=======
mydata = Iterators.repeated(([11.0;2.0],), 1000)
>>>>>>> 091b9bcc1e6b90cf718940f561fdb3b8a01ab123
TH=zeros(4,13);
for epo=1:13
    global plc
    global TH
<<<<<<< HEAD
    Flux.train!((x)->loss_elbo(x[1],x[2]),Flux.params([μ,σ,γ,δ]),mydata,ADAM(0.01))
    plt(θ,α) = pdf.(Normal(μ[1],sqrt(σ[1])),θ) .* pdf.(InverseGamma(γ[1],δ[1]), α)
    contour!(plc,1.3:0.01:4.5,0.0:0.01:28.0,plt,levels=5, linestyle= :dot, linewidth = 3, linecolor = :red)
=======
    Flux.train!((x)->loss_elbo(x[1],x[2]),Flux.params([μ,σ,γ,δ]),mydata,ADAM(0.001))
    plt(θ,α) = pdf(Normal(μ[1],sqrt(σ[1])),θ) .* pdf.(InverseGamma(γ[1],δ[1]), α)
    contour!(plc,10:0.01:14,0:0.1:1.5,plt,levels=5, linestyle=:dash)
>>>>>>> 091b9bcc1e6b90cf718940f561fdb3b8a01ab123
    TH[:,epo].=vcat(μ,σ,γ,δ);
end
TH
gui()
