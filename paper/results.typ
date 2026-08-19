
_Environment: `LunarLanderContinuous-v3`. Run kind: full pre-registered run. N seeds per method per condition: 20. Step budget per training run: 500,000._

== 5.1 Per-method final fitness with BCa 95% CIs

#table(
  columns: (1.6cm, 1.6cm, 1cm, 1.6cm, 1fr),
  align: left,
  stroke: 0.3pt + gray,
  table.header([Method], [Condition], [N], [Mean], [95% BCa CI]),
  [PPO], [clean], [20], [165.1], [[147.4, 184.3]],
  [PPO], [perturbed], [20], [108.1], [[101.0, 114.1]],
  [NEAT], [clean], [20], [34.9], [[-52.5, 103.2]],
  [NEAT], [perturbed], [20], [-37.6], [[-111.7, 18.2]],
)

== 5.2 Pre-registered hypothesis tests (H0-H4)

#table(
  columns: (1.4cm, 1fr, 1.4cm, 1.4cm, 1cm, 1.2cm),
  align: left,
  stroke: 0.3pt + gray,
  table.header([ID], [Test], [p], [alpha], [reject?], [effect]),
  [H0[ppo>random]], [Mann-Whitney U (one-sided, greater)], [0.0000], [0.0250], [yes], [1.00],
  [H0[ppo>heuristic]], [Mann-Whitney U (one-sided, greater)], [1.0000], [0.0250], [no], [-0.95],
  [H0[neat>random]], [Mann-Whitney U (one-sided, greater)], [0.0000], [0.0250], [yes], [0.79],
  [H0[neat>heuristic]], [Mann-Whitney U (one-sided, greater)], [1.0000], [0.0250], [no], [-0.90],
  [H1], [Mann-Whitney U (two-sided)], [0.0144], [0.0125], [no], [-0.46],
  [H2], [Wilcoxon signed-rank (paired, one-sided PPO>NEAT)], [1.0000], [0.0125], [no], [-1.00],
  [H3], [Mann-Whitney U (one-sided, NEAT drop < PPO drop)], [0.9140], [0.0125], [no], [0.25],
)

== 5.3 Compute-fairness sensitivity: FLOPs at matched env-steps

#table(
  columns: (1.6cm, 0.8cm, 1.8cm, 1.8cm, 1.8cm, 1.8cm),
  align: left,
  stroke: 0.3pt + gray,
  table.header([Method], [N], [Forward FLOPs/step], [Inference FLOPs], [Update FLOPs], [Total FLOPs]),
  [PPO], [20], [9.73e+03], [4.88e+09], [1.46e+11], [1.51e+11],
  [NEAT], [20], [1.45e+02], [7.37e+07], [1.01e+06], [7.47e+07],
)
