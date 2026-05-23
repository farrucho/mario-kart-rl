from stable_baselines3.common.callbacks import BaseCallback
import wandb
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os


class WandbTrajectoryCallback(BaseCallback):

    def __init__(
        self,
        reward_log_freq=10_000,
        trajectory_log_freq=200_000,
        verbose=1
    ):
        super().__init__(verbose)

        self.reward_log_freq = reward_log_freq
        self.trajectory_log_freq = trajectory_log_freq

        # latest completed episode
        self.last_finished_episode_info = None

    def _on_step(self) -> bool:

        infos = self.locals["infos"]
        dones = self.locals["dones"]

        # --------------------------------------------------
        # Cache ONE finished episode
        # --------------------------------------------------
        for done, info in zip(dones, infos):

            if done:

                if (
                    "position_history_east" in info
                    and "position_history_south" in info
                ):
                    self.last_finished_episode_info = info
                    break

        # --------------------------------------------------
        # Frequent cheap scalar logging
        # --------------------------------------------------
        if (
            self.num_timesteps % self.reward_log_freq == 0
            and self.last_finished_episode_info is not None
        ):

            self._log_rewards_only(
                self.last_finished_episode_info
            )

        # --------------------------------------------------
        # Expensive trajectory plotting
        # --------------------------------------------------
        if (
            self.num_timesteps % self.trajectory_log_freq == 0
            and self.last_finished_episode_info is not None
        ):

            self._plot_finished_episode(
                self.last_finished_episode_info
            )

        return True

    def _log_rewards_only(self, info):

        reward_total = (
            info["reward_checkpoint"]
            + info["reward_lap"]
            + info["reward_collision"]
            + info["reward_rank"]
            + info["reward_speed"]
            + info["reward_finish_race"]
            + info["reward_time"]
        )

        wandb.log({

            "episode/total_reward":
                reward_total,

            "episode/reward_checkpoint":
                info["reward_checkpoint"],

            "episode/reward_lap":
                info["reward_lap"],

            "episode/reward_collision":
                info["reward_collision"],

            "episode/reward_rank":
                info["reward_rank"],

            "episode/reward_speed":
                info["reward_speed"],

            "episode/reward_finish_race":
                info["reward_finish_race"],

            "episode/reward_time":
                info["reward_time"],

            "episode/final_rank":
                info["rank"],

            "episode/final_lap":
                info["current_lap"]-127,

            "episode/final_checkpoint":
                info["checkpoint"],

            "episode/final_time_seconds":
                info["clock_minutes"] * 60
                + info["clock_seconds"],

            "global_step":
                self.num_timesteps,
        })

    def _plot_finished_episode(self, info):

        east = np.array(info["position_history_east"])
        south = np.array(info["position_history_south"])

        if len(east) < 10:
            return

        # append final position
        east = np.append(
            east,
            info["pos_east"]
        )

        south = np.append(
            south,
            info["pos_south"]
        )

        plt.style.use("dark_background")

        fig, ax = plt.subplots(
            figsize=(14, 14),
            dpi=200
        )

        ax.plot(
            east,
            south,
            linewidth=2,
            alpha=0.9,
        )

        # start point
        ax.scatter(
            east[0],
            south[0],
            s=120,
            marker="o",
            label="Start"
        )

        # end point
        ax.scatter(
            east[-1],
            south[-1],
            s=120,
            marker="X",
            label="End"
        )

        ax.set_aspect("equal")
        ax.margins(0)

        ax.set_xlim(0, 4500)
        ax.set_ylim(0, 4500)

        ax.invert_yaxis()

        ax.grid(True)

        reward_total = (
            info["reward_checkpoint"]
            + info["reward_lap"]
            + info["reward_collision"]
            + info["reward_rank"]
            + info["reward_speed"]
            + info["reward_finish_race"]
            + info["reward_time"]
        )

        ax.set_title(
            (
                f"Finished Episode @ "
                f"{self.num_timesteps:,} timesteps\n"
                f"Reward={reward_total:.2f} | "
                f"Rank={info['rank']/2+1} | "
                f"Lap={info['current_lap']-127}"
            ),
            fontsize=20
        )

        ax.set_xlabel(
            "East",
            fontsize=16
        )

        ax.set_ylabel(
            "South",
            fontsize=16
        )

        ax.tick_params(
            axis='both',
            labelsize=14
        )

        ax.legend(fontsize=12)

        plt.tight_layout()

        # --------------------------------------------------
        # Save image temporarily
        # --------------------------------------------------
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as tmpfile:

            plt.savefig(
                tmpfile.name,
                bbox_inches="tight"
            )

            wandb.log({

                "trajectory/finished_episode":
                    wandb.Image(tmpfile.name),

                "global_step":
                    self.num_timesteps,
            })

            tmp_path = tmpfile.name

        plt.close(fig)

        os.remove(tmp_path)