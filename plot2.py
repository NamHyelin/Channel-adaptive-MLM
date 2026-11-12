import matplotlib.pyplot as plt


# =========================================================
# 1. Temperature / threshold effect
# =========================================================
temp = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

snr_m5 = [0.4020, 0.4350, 0.4591, 0.4690, 0.4760, 0.4805, 0.4842, 0.4866, 0.4868, 0.4870]
# 0.4020, 0.4350, 0.4591, 0.4660, 0.4690, 0.4705, 0.4712, 0.4716, 0.4718, 0.4720

snr_0 = [0.5951, 0.6200, 0.6380, 0.6490, 0.6550, 0.6585, 0.6605, 0.6617, 0.6624, 0.6630]

snr_5 = [0.9651, 0.9685, 0.9707, 0.9730, 0.9750, 0.9765, 0.9775, 0.9782, 0.9786, 0.9788]


plt.figure(figsize=(5, 4.5))

plt.plot(
    temp,
    snr_m5,
    marker="o",
    linestyle="-",
    markerfacecolor="none",
    markersize=10,
    color="blue",
    label="SNR = -5 dB",
)

plt.plot(
    temp,
    snr_0,
    marker="s",
    linestyle="--",
    markerfacecolor="none",
    markersize=10,
    color="orange",
    label="SNR = 0 dB",
)

plt.plot(
    temp,
    snr_5,
    marker="^",
    linestyle="-.",
    markerfacecolor="none",
    markersize=10,
    color="tab:cyan",
    label="SNR = 5 dB",
)

plt.xlabel("Masking threshold, $\\tau$", fontsize=15)
plt.ylabel("Cosine similarity", fontsize=15)

plt.xticks(temp, fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(0.5, 5)
plt.ylim(0.4, 1.01)

plt.grid(True, alpha=0.3)

plt.legend(fontsize=12, loc=[0.05, 0.62], frameon=True, framealpha=0.5)

plt.tight_layout()


plt.savefig(
    "C:\\token_bit_transformer\\results\\params\\threshold.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# =========================================================
# 2. Gate effect
# =========================================================
snr = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

g0 = [
    0.4128,
    0.4148,
    0.4191,
    0.4337,
    0.4689,
    0.5465,
    0.6520,
    0.7723,
    0.8646,
    0.9230,
    0.9630,
    0.9854,
    0.9949,
    0.9991,
    0.9997,
    1.0000,
]

g025 = [
    0.4248,
    0.4374,
    0.4611,
    0.4985,
    0.5558,
    0.6442,
    0.7360,
    0.8299,
    0.8967,
    0.9406,
    0.9707,
    0.9883,
    0.9959,
    0.9992,
    0.9997,
    1.0000,
]

g2 = [
    0.4605,
    0.4697,
    0.4851,
    0.5103,
    0.5483,
    0.6115,
    0.6944,
    0.7914,
    0.8709,
    0.9266,
    0.9664,
    0.9869,
    0.9952,
    0.9992,
    0.9997,
    1.0000,
]

exp_ber = [
    0.4631,
    0.4738,
    0.4913,
    0.5224,
    0.5712,
    0.6473,
    0.7363,
    0.8304,
    0.8978,
    0.9410,
    0.9707,
    0.9883,
    0.9959,
    0.9992,
    0.9997,
    1.0000,
]


plt.figure(figsize=(6, 4.5))

plt.plot(
    snr,
    exp_ber,
    marker="o",
    linestyle="-",
    linewidth=1,
    markerfacecolor="none",
    markersize=7,
    color="blue",
    label="Exp-BER",
)


plt.plot(
    snr,
    g025,
    marker="^",
    linestyle="--",
    linewidth=1,
    markerfacecolor="none",
    markersize=7,
    color="tab:cyan",
    label="$\omega$=2",
)


plt.plot(
    snr,
    g2,
    marker="*",
    linestyle="-.",
    linewidth=1,
    markerfacecolor="none",
    markersize=7,
    color="orange",
    label="$\omega$=0.25",
)

plt.plot(
    snr,
    g0,
    marker="D",
    linestyle=":",
    linewidth=1,
    markerfacecolor="none",
    markersize=7,
    color="tab:green",
    label="$\omega$=0",
)


plt.xlabel("SNR (dB)", fontsize=15)
plt.ylabel('Cosine similarity', fontsize=15)

plt.xticks(snr, fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(-5, 10)
plt.ylim(0.4, 1.01)

plt.grid(True, alpha=0.3)

# plt.tick_params(
#     axis="y",
#     which="both",
#     # left=False,
#     right=False,
#     labelleft=False,
#     labelright=False,
# )

plt.legend(fontsize=12, loc=[0.05, 0.62], frameon=True, framealpha=0.5)

plt.tight_layout()

plt.savefig(
    "C:\\token_bit_transformer\\results\\params\\gate.png", dpi=300, bbox_inches="tight"
)

plt.show()
plt.close()
