# Every observation carries device time, host receive time, age and validity.
builtin_interfaces/Time source_time
builtin_interfaces/Time host_receive_time
builtin_interfaces/Time acquisition_start
builtin_interfaces/Time acquisition_end
uint64 source_sequence
bool valid
builtin_interfaces/Duration age
string invalid_reason
string source_clock_domain
string host_clock_domain
builtin_interfaces/Time mapped_host_time
bool timing_valid
