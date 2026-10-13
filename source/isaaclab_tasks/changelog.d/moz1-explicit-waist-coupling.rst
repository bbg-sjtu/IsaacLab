Changed
^^^^^^^

* Exposed the Moz1 waist sum preference through ``JointCouplingTaskCfg`` in
  ``controller.fixed_input_tasks``. Adjust its ``cost`` instead of
  ``waist_coupling_cost``, which is now zero in the task configuration to avoid
  duplicate weighting. Preserved the separate joint-sign preferences.
