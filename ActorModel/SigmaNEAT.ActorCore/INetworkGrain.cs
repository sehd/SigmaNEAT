using Orleans;
using System.Threading.Tasks;

namespace SigmaNEAT.ActorCore
{
    public interface INetworkGrain : IGrainWithIntegerKey
    {
        Task<double> GetOutput(double[] input);
    }
}
