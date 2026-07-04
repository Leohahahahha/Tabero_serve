from enum import Enum, auto


class TactileType(Enum):
    """
    Tactile (force / torque) usage modes.

    Current token layout (only the expert history + future joint prediction path is retained):

        |<------- prefix (LLM) ---->|<----------- suffix (expert) --------------------->|
        |<- images ->|<- language ->|<- tactile(hist) ->|<- state ->|<- actions+tactile ->|

    - prefix: contains only vision + language, with no tactile-related tokens.
    - suffix: inserts an expert token, produced from multi-frame historical tactile data via concat+MLP, before the state;
      then comes the state token, followed by a full [action_dim]-dimensional action sequence whose last tactile_dim dimensions are force/torque.
    """

    NO = auto()
    """Do not use tactile data. It may still remain in the data for statistics, but the model ignores it completely."""

    EXPERT_HIS_C_FUT = auto()
    """The only currently retained tactile/force mode:

    - Input side: concatenate multi-frame tactile history into one vector and project it through an MLP into one token as the expert condition (HIS_C).
    - Output side: the decoder learns [actions + tactile force] on the action channel, with separate weighted supervision for actions and tactile force inside the loss (FUT).
    """


