# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.legged_env import LeggedEnv
from legged_lab.envs.base.tt_env import TTEnv
from legged_lab.envs.base.tt_env_mp import TTEnvMP

from legged_lab.envs.t1_tt.t1_tt_config import (
    T1TableTennisEnvCfg,
    T1TableTennisAgentCfg,
    T1TT_EvalEnvCfg,
    T1TableTennisMPEnvCfg,
    T1TT_MP_EvalEnvCfg,
    T1TableTennisMPAgentCfg,
    T1TableTennisMPGEnvCfg,
    T1TT_MPG_EvalEnvCfg,
    T1TableTennisMPGAgentCfg,
    T1TT_MPG_EvalLongCfg,
    T1TT_MPG_EvalMidCfg,
    T1TT_MPG_EvalShortCfg,
)


from legged_lab.utils.task_registry import task_registry
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg()) #TTEnv
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg())
task_registry.register("t1_tt_mp", TTEnvMP, T1TableTennisMPEnvCfg(), T1TableTennisMPAgentCfg())
task_registry.register("t1_tt_mp_eval", TTEnvMP, T1TT_MP_EvalEnvCfg(), T1TableTennisMPAgentCfg())
task_registry.register("t1_tt_mpg", TTEnvMP, T1TableTennisMPGEnvCfg(), T1TableTennisMPGAgentCfg())
task_registry.register("t1_tt_mpg_eval", TTEnvMP, T1TT_MPG_EvalEnvCfg(), T1TableTennisMPGAgentCfg())
task_registry.register("t1_tt_mpg_eval_long", TTEnvMP, T1TT_MPG_EvalLongCfg(), T1TableTennisMPGAgentCfg())
task_registry.register("t1_tt_mpg_eval_mid", TTEnvMP, T1TT_MPG_EvalMidCfg(), T1TableTennisMPGAgentCfg())
task_registry.register("t1_tt_mpg_eval_short", TTEnvMP, T1TT_MPG_EvalShortCfg(), T1TableTennisMPGAgentCfg())
