"""
The main purpose behind this is to move the update sims to be before 
save_default_domain

The reason for this is that super().reset() closes all the sim objects and then
when you call save_default_domain, which requires those objects not to have been 
closed, it errors. 
"""
from robosuite.wrappers import Wrapper, DomainRandomizationWrapper


class FixedDomainRandomizationWrapper(DomainRandomizationWrapper):


    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = Wrapper.reset(self)

        # update sims
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0


        if self.randomize_on_reset:
            # domain randomize + regenerate observation
            self.randomize_domain()
            ret = self.env._get_observations(force_update=True)

        return ret

