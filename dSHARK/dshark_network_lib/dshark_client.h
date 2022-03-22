#pragma once
#include "dshark_common.h"
#include "dshark_message.h"
#include "dshark_queue.h"
#include "dshark_connection.h"


namespace dshark {
	template <typename T>
	class client_interface
	{
	public:
		client_interface() : m_socket(m_context)
		{

		}

		virtual ~client_interface()
		{
			Disconnect();
		}
	public:
		bool Connect(const std::string& host, const uint16_t port)
		{
			try
			{
				// Resolve hostname/ip-address into tangiable physical address
				asio::ip::tcp::resolver resolver(m_context);
				asio::ip::tcp::resolver::results_type endpoints = resolver.resolve(host, std::to_string(port));

				// Create connection
				m_connection = std::make_unique<connection<T>>(connection<T>::owner::client, m_context, asio::ip::tcp::socket(m_context), m_qMessagesIn);

				// Tell the connection object to connect to server
				m_connection->ConnectToServer(endpoints);

				// Start Context Thread
				thrContext = std::thread([this]() { m_context.run(); });
			}
			catch (std::exception& e)
			{
				std::cerr << "Client Exception: " << e.what() << "\n";
				return false;
			}
			return true;
		}
		void Disconnect()
		{
			// If connection exists, and it's connected then...
			if (IsConnected())
			{
				// ...disconnect from server gracefully
				m_connection->Disconnect();
			}

			// Either way, we're also done with the asio context...				
			m_context.stop();
			// ...and its thread
			if (thrContext.joinable())
				thrContext.join();

			// Destroy the connection object
			m_connection.release();
		}
		bool IsConnected()
		{
			if (m_connection) {
				return m_connection->IsConnected();
			}
			else {
				return false;
			}
		}
		void Send(const message<T>& msg)
		{
			if (IsConnected())
				m_connection->Send(msg);
		}

		// Retrieve queue of messages from server
		dsqueue<owned_message<T>>& Incoming()
		{
			return m_qMessagesIn;
		}

	protected:
		asio::io_context m_context;
		std::thread thrContext;
		asio::ip::tcp::socket m_socket;
		std::unique_ptr<connection<T>> m_connection;

	private:
		dsqueue<owned_message<T>> m_qMessagesIn;

	};
}
