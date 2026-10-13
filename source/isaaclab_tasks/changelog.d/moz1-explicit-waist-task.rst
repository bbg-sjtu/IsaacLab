Changed
^^^^^^^

* Moved all Moz1 waist preferences into ``controller.fixed_input_tasks`` using
  ``WaistTaskCfg`` and the standard Pink controller. Remove that task to disable
  waist preferences; no tasks are appended internally.

Deprecated
^^^^^^^^^^

* Deprecated action-level ``waist_coupling_cost`` and ``waist_sign_cost`` in favor
  of ``WaistTaskCfg.cost`` and ``WaistTaskCfg.sign_cost``. Legacy overrides now
  require an explicitly configured waist task instead of creating one implicitly.
