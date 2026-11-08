"""Donor notes for the shuffled-recap control (arm `recap_shuffled`).

Empty until the recap arm has run: the operator harvests the `note` calls of the recap-arm traces
with memory.workmem.donors_from_traces and overwrites this file with
memory.workmem.write_donor_module, before building the control. With no donors, the control's
recap says it is unavailable; check memory.workmem.control_ready() before building the control."""
DONORS = ()
